"""
OPTIONAL BONUS ABLATION — Attention analysis: Model H (with PE) vs (no PE).

Produces side-by-side heatmaps for the same input sentence with both models,
plus entropy and mean-distance bar charts comparing every layer/head.

Usage (from lm_project/):
    python phase2/ablation/attention_nope.py

Outputs (in report/):
    ablation_attn_layer{00,06,11}.png   — side-by-side heatmaps
    ablation_attn_entropy.png           — entropy bar chart per layer
    ablation_attn_distance.png          — mean attention distance bar chart
    ablation_attn_summary.json          — numeric entropy/distance values
"""

import os, sys, json
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "phase2"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sentencepiece as spm

EXAMPLE_SENTENCE = (
    "भारत एशिया का एक प्रमुख देश है। "
    "यहाँ की संस्कृति और परंपराएँ बहुत पुरानी हैं।"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_attention(model, token_ids, device):
    """Run one forward pass; return list of (h, T, T) attention weight tensors."""
    model.eval()
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    model(idx)
    return [block.attn.last_attn_weights[0].cpu().float()
            for block in model.blocks]


def attn_entropy(weights):
    """Mean per-head entropy: − Σ p log p, averaged over heads and query positions."""
    eps = 1e-9
    H   = -(weights * (weights + eps).log())   # (h, T, T)
    return H.sum(-1).mean().item()             # scalar


def mean_attn_distance(weights):
    """Average distance (in token positions) each query attends to."""
    h, T, _ = weights.shape
    pos = torch.arange(T, dtype=torch.float32)
    dist = 0.0
    for q in range(T):
        row  = weights[:, q, :q+1]              # (h, q+1) — causal, can only look back
        kpos = pos[:q+1].unsqueeze(0)            # (1, q+1)
        dist += (row * (q - kpos)).sum().item()
    return dist / (h * T)


def load_model(ModelClass, cfg, ckpt_rel):
    """Load a model from a relative checkpoint path."""
    from model    import DecoderLM       # noqa: F401 (unused if ModelClass != DecoderLM)
    ckpt_path = os.path.join(BASE_DIR, ckpt_rel)
    if not os.path.exists(ckpt_path):
        print(f"  [warn] checkpoint not found: {ckpt_path}")
        return None
    m  = ModelClass(cfg)
    ck = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(ck["model"])
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap_pair(attn_pe, attn_npe, labels, layer_idx, out_dir):
    """Side-by-side attention heatmap: with PE (left) vs no PE (right)."""
    T   = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Attention Weights — Layer {layer_idx:02d}, Head 0\n"
                 f"Model H with PE (left) vs without PE (right)",
                 fontsize=12, fontweight="bold")

    for ax, data, title, color in [
        (axes[0], attn_pe [layer_idx][0].numpy()[:T, :T], "Model H (with PE)",  "viridis"),
        (axes[1], attn_npe[layer_idx][0].numpy()[:T, :T], "Model H (no PE)",    "plasma"),
    ]:
        sns.heatmap(data, ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    cmap=color, vmin=0.0, cbar=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Key position (attended to)", fontsize=10)
        ax.set_ylabel("Query position (attending)", fontsize=10)
        ax.tick_params(axis="x", labelsize=6, rotation=45)
        ax.tick_params(axis="y", labelsize=6)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"ablation_attn_layer{layer_idx:02d}.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {fname}")


def plot_entropy_bars(ent_pe, ent_npe, out_dir):
    """Bar chart: attention entropy per layer for both models."""
    n_layers = len(ent_pe)
    x = np.arange(n_layers)
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, n_layers), 5))
    ax.bar(x - w/2, ent_pe,  w, color="#1e50a0", label="Model H (with PE)")
    ax.bar(x + w/2, ent_npe, w, color="#c03000", label="Model H (no PE)", alpha=0.85)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Mean attention entropy (nats)", fontsize=11)
    ax.set_title("Attention Entropy per Layer — Ablation Comparison", fontsize=12,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_layers)])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir, "ablation_attn_entropy.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {fname}")


def plot_distance_bars(dist_pe, dist_npe, out_dir):
    """Bar chart: mean attention distance per layer for both models."""
    n_layers = len(dist_pe)
    x = np.arange(n_layers)
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, n_layers), 5))
    ax.bar(x - w/2, dist_pe,  w, color="#1e50a0", label="Model H (with PE)")
    ax.bar(x + w/2, dist_npe, w, color="#c03000", label="Model H (no PE)", alpha=0.85)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Mean attention distance (tokens)", fontsize=11)
    ax.set_title("Mean Attention Distance per Layer — Ablation Comparison",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_layers)])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir, "ablation_attn_distance.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    from model           import DecoderLM
    from config_H        import CONFIG_H
    from model_nope      import DecoderLM_NoPE
    from config_ablation import ABLATION_CONFIG

    tok_path = os.path.join(BASE_DIR, CONFIG_H.tokenizer_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)

    token_ids = [sp.bos_id()] + sp.EncodeAsIds(EXAMPLE_SENTENCE)
    token_ids = token_ids[:CONFIG_H.context_len]
    T         = len(token_ids)
    pieces    = [sp.IdToPiece(t)[:6] for t in token_ids]   # truncate for display

    report_dir = os.path.join(BASE_DIR, "report")
    os.makedirs(report_dir, exist_ok=True)

    # Load both models
    model_pe  = load_model(DecoderLM,     CONFIG_H,       "phase2/checkpoints_H/best.pt")
    model_npe = load_model(DecoderLM_NoPE, ABLATION_CONFIG, "phase2/ablation/checkpoints/best.pt")

    if model_pe  is None: print("[error] Model H checkpoint missing"); return
    if model_npe is None: print("[error] NoPE checkpoint missing — train first"); return

    model_pe  = model_pe.to(device)
    model_npe = model_npe.to(device)

    attn_pe  = get_attention(model_pe,  token_ids, device)
    attn_npe = get_attention(model_npe, token_ids, device)

    n_layers = len(attn_pe)

    # ── Heatmaps for layer 0, middle, last ───────────────────────────────────
    for layer_idx in [0, n_layers // 2, n_layers - 1]:
        plot_heatmap_pair(attn_pe, attn_npe, pieces, layer_idx, report_dir)

    # ── Entropy + distance per layer ─────────────────────────────────────────
    ent_pe   = [attn_entropy(attn_pe[l])        for l in range(n_layers)]
    ent_npe  = [attn_entropy(attn_npe[l])       for l in range(n_layers)]
    dist_pe  = [mean_attn_distance(attn_pe[l])  for l in range(n_layers)]
    dist_npe = [mean_attn_distance(attn_npe[l]) for l in range(n_layers)]

    plot_entropy_bars(ent_pe, ent_npe, report_dir)
    plot_distance_bars(dist_pe, dist_npe, report_dir)

    summary = {
        "example": EXAMPLE_SENTENCE[:60],
        "n_tokens": T,
        "entropy_with_pe":  [round(e, 4) for e in ent_pe],
        "entropy_no_pe":    [round(e, 4) for e in ent_npe],
        "distance_with_pe": [round(d, 4) for d in dist_pe],
        "distance_no_pe":   [round(d, 4) for d in dist_npe],
        "avg_entropy_with_pe":  round(sum(ent_pe)  / n_layers, 4),
        "avg_entropy_no_pe":    round(sum(ent_npe)  / n_layers, 4),
        "avg_distance_with_pe": round(sum(dist_pe) / n_layers, 4),
        "avg_distance_no_pe":   round(sum(dist_npe) / n_layers, 4),
    }
    out_path = os.path.join(report_dir, "ablation_attn_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_path}")

    # Print summary table
    print(f"\n{'Metric':<35} {'With PE':>10} {'No PE':>10} {'Δ':>10}")
    print("-" * 68)
    print(f"  {'Avg entropy (all layers)':<33} "
          f"{summary['avg_entropy_with_pe']:>10.4f} "
          f"{summary['avg_entropy_no_pe']:>10.4f} "
          f"{summary['avg_entropy_no_pe']-summary['avg_entropy_with_pe']:>+10.4f}")
    print(f"  {'Avg distance (all layers)':<33} "
          f"{summary['avg_distance_with_pe']:>10.4f} "
          f"{summary['avg_distance_no_pe']:>10.4f} "
          f"{summary['avg_distance_no_pe']-summary['avg_distance_with_pe']:>+10.4f}")


if __name__ == "__main__":
    main()
