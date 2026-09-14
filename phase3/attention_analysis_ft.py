"""
Attention analysis: pretrained vs. fine-tuned models on reasoning inputs.

For each model (H and L) produces side-by-side attention heatmaps showing
how the model attends to tokens in a reasoning example before and after
fine-tuning.  Comparison reveals whether fine-tuning sharpens attention on
the relevant entity / comparison words.

Outputs
-------
  report/attn_ft_Model_H_layerXX.png
  report/attn_ft_Model_L_layerXX.png
  report/attn_ft_summary.json

Usage
-----
    python phase3/attention_analysis_ft.py --model H
    python phase3/attention_analysis_ft.py --model L
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sentencepiece as spm


PROMPT_PREFIX = "प्रश्न: "
ANSWER_PREFIX = "\nउत्तर: "

# Representative reasoning examples per language
HINDI_EXAMPLES = [
    "राम की उम्र 25 साल है। श्याम की उम्र 30 साल है। किसकी उम्र अधिक है?",
    "अर्जुन की ऊँचाई सोहन से अधिक है। सोहन की ऊँचाई विक्रम से अधिक है। सबसे कम ऊँचाई किसकी है?",
]
NEPALI_EXAMPLES = [
    "सन्तोष को उमेर ३५ वर्ष छ। मनिता को उमेर ४२ वर्ष छ। कसको उमेर बढी छ?",
    "राजेन्द्र को उचाइ सुमित्रा भन्दा बढी छ। सुमित्रा को उचाइ गोविन्द भन्दा बढी छ। सबैभन्दा कम उचाइ भएको को हो?",
]


# ── Attention extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def get_attention(model, token_ids, device):
    """Run forward pass; extract attention weights (one tensor per layer)."""
    model.eval()
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    model(idx)
    return [block.attn.last_attn_weights[0].cpu().float()  # (h, T, T)
            for block in model.blocks]


def load_model(ckpt_path, cfg, device):
    from phase2.model import DecoderLM
    model = DecoderLM(cfg).to(device)
    ck    = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    return model


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(attn_pre, attn_ft, labels, layer_idx, model_name, out_dir):
    """
    Side-by-side heatmap: pretrained (left) vs. fine-tuned (right).
    Shows head 0 of the specified layer for both models.
    """
    T = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{model_name} — Layer {layer_idx:02d}, Head 0", fontsize=13)

    for ax, data, title in [
        (axes[0], attn_pre[layer_idx][0].numpy()[:T, :T], "Pretrained"),
        (axes[1], attn_ft [layer_idx][0].numpy()[:T, :T], "Fine-tuned"),
    ]:
        sns.heatmap(data, ax=ax, xticklabels=labels, yticklabels=labels,
                    cmap="viridis", vmin=0.0, cbar=True)
        ax.set_title(title)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.tick_params(axis="x", labelsize=6, rotation=45)
        ax.tick_params(axis="y", labelsize=6)

    plt.tight_layout()
    fname = f"attn_ft_{model_name}_layer{layer_idx:02d}.png"
    path  = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def attn_entropy(weights):
    """Mean attention entropy across heads (higher = more diffuse)."""
    eps = 1e-9
    H   = -(weights * (weights + eps).log()).sum(dim=-1)  # (h, T)
    return H.mean().item()


# ── Main ──────────────────────────────────────────────────────────────────────

def run(model_key, base_dir):
    if model_key == "H":
        from config_ft_H import CONFIG_FT_H as cfg
        pretrained_ckpt = os.path.join(base_dir, "phase2/checkpoints_H/best.pt")
        finetuned_ckpt  = os.path.join(base_dir, "phase3/checkpoints_ft_H/best_ft.pt")
        examples        = HINDI_EXAMPLES
    else:
        from config_ft_L import CONFIG_FT_L as cfg
        pretrained_ckpt = os.path.join(base_dir, "phase2/checkpoints_L/best.pt")
        finetuned_ckpt  = os.path.join(base_dir, "phase3/checkpoints_ft_L/best_ft.pt")
        examples        = NEPALI_EXAMPLES

    if not os.path.exists(finetuned_ckpt):
        print(f"[warn] Finetuned checkpoint not found: {finetuned_ckpt}  — skipping.")
        return {}

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tok_path = os.path.join(base_dir, cfg.tokenizer_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)

    bos_id = sp.bos_id()

    report_dir = os.path.join(base_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    pre_model = load_model(pretrained_ckpt, cfg, device)
    ft_model  = load_model(finetuned_ckpt,  cfg, device)

    layer_indices = [0, cfg.n_layers // 2, cfg.n_layers - 1]
    summary = []

    for i, text in enumerate(examples):
        full_text = f"{PROMPT_PREFIX}{text}{ANSWER_PREFIX}"
        token_ids = [bos_id] + sp.EncodeAsIds(full_text)
        # Truncate to context window
        token_ids = token_ids[:cfg.context_len]
        T         = len(token_ids)

        pieces   = [sp.IdToPiece(t) for t in token_ids]
        # Shorten pieces for display
        labels   = [p[:6] if len(p) > 6 else p for p in pieces]

        attn_pre = get_attention(pre_model, token_ids, device)
        attn_ft  = get_attention(ft_model,  token_ids, device)

        for layer_idx in layer_indices:
            plot_comparison(attn_pre, attn_ft, labels, layer_idx,
                            cfg.model_name, report_dir)

        # Entropy comparison
        pre_ents = [attn_entropy(attn_pre[l]) for l in layer_indices]
        ft_ents  = [attn_entropy(attn_ft [l]) for l in layer_indices]
        summary.append({
            "example_idx":    i,
            "example_prefix": text[:50],
            "pre_entropy":    {f"layer{l}": round(e, 4)
                               for l, e in zip(layer_indices, pre_ents)},
            "ft_entropy":     {f"layer{l}": round(e, 4)
                               for l, e in zip(layer_indices, ft_ents)},
        })

    del pre_model, ft_model

    out = {
        "model_name":     cfg.model_name,
        "layer_analysed": layer_indices,
        "examples":       summary,
    }
    out_path = os.path.join(report_dir, f"attn_ft_{cfg.model_name}_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["H", "L"], action="append", dest="models")
    args = parser.parse_args()
    if not args.models:
        args.models = ["H", "L"]

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    for m in args.models:
        run(m, base_dir)
