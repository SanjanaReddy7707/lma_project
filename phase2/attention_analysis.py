"""
Phase 2 — Attention Analysis.

For each model produces:
  1. Attention heatmaps: early layer (0), middle, late (N-1) × multiple heads
  2. Per-head attention entropy (higher = more diffuse)
  3. Mean attention distance per head/layer (local vs. long-range)
  4. Summary statistics saved to JSON + heatmap PNGs in report/

Usage:
    python phase2/attention_analysis.py --model H
    python phase2/attention_analysis.py --model L
"""

import os, sys, json, math, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_attention_weights(model, token_ids, device):
    """
    Run a forward pass and collect attention weight matrices from every block.
    Returns list of tensors, one per layer: shape (h, T, T).
    (The model stores last_attn_weights in each CausalSelfAttention after forward.)
    """
    model.eval()
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    model(idx)  # forward pass populates last_attn_weights in each block
    attn_per_layer = []
    for block in model.blocks:
        # shape: (1, h, T, T) → (h, T, T)
        w = block.attn.last_attn_weights[0].cpu().float()
        attn_per_layer.append(w)
    return attn_per_layer  # list[Tensor(h, T, T)], length = n_layers


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmap(attn_weights, token_labels, layer_idx, model_name,
                            out_dir, max_heads=4):
    """
    Plot attention heatmap for one layer showing up to max_heads heads.
    attn_weights: (h, T, T) tensor
    """
    h, T, _ = attn_weights.shape
    n_heads  = min(h, max_heads)
    fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 5))
    if n_heads == 1:
        axes = [axes]

    labels = token_labels[:T]

    for head_idx, ax in enumerate(axes[:n_heads]):
        data = attn_weights[head_idx].numpy()  # (T, T)
        sns.heatmap(
            data, ax=ax,
            xticklabels=labels, yticklabels=labels,
            cmap="viridis", vmin=0.0,
            cbar=(head_idx == n_heads - 1),
        )
        ax.set_title(f"Head {head_idx}", fontsize=10)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.tick_params(axis='x', labelsize=6, rotation=45)
        ax.tick_params(axis='y', labelsize=6, rotation=0)

    fig.suptitle(f"{model_name} — Layer {layer_idx} Attention Weights",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(out_dir, f"attn_{model_name}_layer{layer_idx:02d}.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Entropy and distance
# ─────────────────────────────────────────────────────────────────────────────

def attention_entropy(attn_weights):
    """
    Per-head average entropy of attention distributions.
    attn_weights: (h, T, T)
    Returns: (h,) array of mean entropy values.
    """
    eps = 1e-9
    # entropy per query position: -sum(p * log(p)) over key axis
    # shape: (h, T, T) → (h, T)
    ent = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
    # Average over T query positions → (h,)
    return ent.mean(dim=-1).numpy()


def mean_attention_distance(attn_weights):
    """
    Average distance (|query_pos - key_pos|) each head attends to.
    Higher = more long-range; lower = more local.
    attn_weights: (h, T, T)
    Returns: (h,) array of mean distances.
    """
    h, T, _ = attn_weights.shape
    positions = torch.arange(T, dtype=torch.float32)
    # dist[q, k] = |q - k|
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # (T, T)
    dist = dist.unsqueeze(0).expand(h, -1, -1)  # (h, T, T)
    weighted_dist = (attn_weights * dist).sum(dim=-1)  # (h, T)
    return weighted_dist.mean(dim=-1).numpy()  # (h,)


# ─────────────────────────────────────────────────────────────────────────────
# Summary plot: entropy and distance per layer/head
# ─────────────────────────────────────────────────────────────────────────────

def plot_entropy_distance(all_entropy, all_distance, model_name, out_dir):
    """
    all_entropy: list[array(h,)] one per layer
    all_distance: list[array(h,)] one per layer
    """
    n_layers = len(all_entropy)
    n_heads  = len(all_entropy[0])

    # Shape: (n_layers, n_heads)
    ent_mat  = np.stack(all_entropy)
    dist_mat = np.stack(all_distance)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, cbar_label in [
        (axes[0], ent_mat,  "Attention Entropy (nats)", "entropy"),
        (axes[1], dist_mat, "Mean Attention Distance (tokens)", "distance"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap="plasma", origin="upper")
        ax.set_xlabel("Head index", fontsize=11)
        ax.set_ylabel("Layer index", fontsize=11)
        ax.set_title(f"{model_name} — {title}", fontsize=12, fontweight="bold")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        plt.colorbar(im, ax=ax, label=cbar_label)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"attn_summary_{model_name}.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_attention_analysis(cfg, base_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    from model import DecoderLM
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(base_dir, cfg.tokenizer_path))

    model = DecoderLM(cfg).to(device)
    ckpt_path = os.path.join(base_dir, cfg.checkpoint_dir, "best.pt")
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"[{cfg.model_name}] loaded checkpoint step={ck.get('step','?')}")

    # ── Pick an example sentence from test split ──────────────────────────
    test_path = os.path.join(base_dir, "data", cfg.language, "splits", "test.txt")
    example_sentence = None
    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if 20 <= len(line) <= 200:
                    example_sentence = line
                    break

    if example_sentence is None:
        example_sentence = "यह एक परीक्षण वाक्य है।" if cfg.language == "hindi" \
                           else "यो एक परीक्षण वाक्य हो।"

    token_ids = sp.EncodeAsIds(example_sentence)
    T = min(len(token_ids), 32)  # cap at 32 for readable heatmaps
    token_ids = token_ids[:T]
    token_labels = [sp.IdToPiece(tid) for tid in token_ids]

    print(f"\nExample sentence ({T} tokens):")
    print(f"  {example_sentence[:120]}")
    print(f"  tokens: {token_labels}\n")

    # ── Extract attention weights ─────────────────────────────────────────
    attn_per_layer = get_attention_weights(model, token_ids, device)

    # ── Heatmaps for early, middle, late layers ───────────────────────────
    n_layers      = len(attn_per_layer)
    layer_indices = sorted({0, n_layers // 2, n_layers - 1})
    print("Generating heatmaps...")
    for li in layer_indices:
        plot_attention_heatmap(
            attn_per_layer[li][:, :T, :T],
            token_labels, li, cfg.model_name, out_dir, max_heads=4,
        )

    # ── Entropy and distance per layer/head ───────────────────────────────
    all_entropy, all_distance = [], []
    for layer_attn in attn_per_layer:
        w = layer_attn[:, :T, :T]
        all_entropy.append(attention_entropy(w))
        all_distance.append(mean_attention_distance(w))

    print("\nGenerating entropy/distance summary plot...")
    plot_entropy_distance(all_entropy, all_distance, cfg.model_name, out_dir)

    # ── Print per-layer summary table ─────────────────────────────────────
    print(f"\n{'Layer':<6} {'Head':<5} {'Entropy':>10} {'MeanDist':>10}")
    print("-" * 35)
    for li, (ent_arr, dist_arr) in enumerate(zip(all_entropy, all_distance)):
        for hi, (ent, dist) in enumerate(zip(ent_arr, dist_arr)):
            print(f"  {li:<4} {hi:<4} {ent:>10.4f} {dist:>10.4f}")

    # ── Save JSON summary ─────────────────────────────────────────────────
    summary = {
        "model": cfg.model_name,
        "language": cfg.language,
        "example_sentence": example_sentence,
        "token_labels": token_labels,
        "n_layers": n_layers,
        "n_heads": cfg.n_heads,
        "entropy_per_layer_head": [e.tolist() for e in all_entropy],
        "distance_per_layer_head": [d.tolist() for d in all_distance],
    }
    json_path = os.path.join(out_dir, f"attn_summary_{cfg.model_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {json_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["H", "L"], required=True)
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    out_dir  = os.path.join(base_dir, "report")

    if args.model == "H":
        from config_H import CONFIG_H as cfg
    else:
        from config_L import CONFIG_L as cfg

    run_attention_analysis(cfg, base_dir, out_dir)
