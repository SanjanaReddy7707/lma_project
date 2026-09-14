"""
OPTIONAL BONUS ABLATION — Evaluation for Model H vs Model H (no PE).

Runs the full Phase 2 eval suite on both checkpoints and writes a
side-by-side comparison JSON plus a printed table.

Usage (from lm_project/):
    python phase2/ablation/eval_nope.py

Outputs:
    report/ablation_eval.json      — full numeric results
    report/ablation_curves.png     — training loss curves side-by-side
"""

import os, sys, math, json
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "phase2"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import sentencepiece as spm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers  (self-contained copies so no import from eval_metrics.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bpb(loss_nats, tokenizer, sample_token_ids):
    """BPB = cross-entropy (nats/token) / (ln2 × avg_utf8_bytes_per_token)."""
    total_bytes  = sum(len(tokenizer.IdToPiece(tid).encode("utf-8"))
                       for seq in sample_token_ids for tid in seq)
    total_tokens = sum(len(seq) for seq in sample_token_ids)
    avg_bytes    = total_bytes / max(total_tokens, 1)
    return loss_nats / (math.log(2) * avg_bytes)


def repetition_rate(token_ids_list, n=4):
    """Fraction of 4-grams in generated output that are repeated."""
    total, repeated = 0, 0
    for ids in token_ids_list:
        ngrams  = [tuple(ids[i:i+n]) for i in range(len(ids) - n + 1)]
        total  += len(ngrams)
        counts  = Counter(ngrams)
        repeated += sum(v - 1 for v in counts.values())
    return repeated / max(total, 1)


def distinct_n(token_ids_list, n):
    """Unique n-grams / total n-grams across all generated sequences."""
    all_ng, unique = [], set()
    for ids in token_ids_list:
        for i in range(len(ids) - n + 1):
            g = tuple(ids[i:i+n])
            all_ng.append(g)
            unique.add(g)
    return len(unique) / max(len(all_ng), 1)


def compute_rouge_l(hyp, ref):
    """ROUGE-L F1 via longest common subsequence (token lists)."""
    m, n = len(ref), len(hyp)
    if not m or not n:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if ref[i-1] == hyp[j-1] \
                       else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p, r = lcs / n, lcs / m
    return (2 * p * r / (p + r)) if (p + r) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PPL / BPB
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_ppl_bpb(model, loader, device, tokenizer, max_batches=200):
    """Return (avg_loss, PPL, BPB)."""
    model.eval()
    total_loss, n = 0.0, 0
    sample_ids = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        _, loss = model(x.to(device), y.to(device))
        total_loss += loss.item()
        n += 1
        if i < 10:
            sample_ids.extend(x.cpu().tolist())
    avg_loss = total_loss / max(n, 1)
    return avg_loss, math.exp(avg_loss), compute_bpb(avg_loss, tokenizer, sample_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_temp(model, prompt_ids, max_new, temp, device, cfg):
    """Generate max_new tokens from prompt at temperature temp."""
    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        idx_c = idx[:, -cfg.context_len:]
        logits, _ = model(idx_c)
        logits = logits[:, -1, :]
        if temp == 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temp, dim=-1)
            nxt   = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx[0, len(prompt_ids):].tolist()


def eval_generation(model, sp, test_lines, cfg, device,
                    n_samples=100, prefix_tok=64, gen_tok=50):
    """Run generation eval at greedy / t=0.5 / t=1.0 / t=1.5."""
    try:
        import sacrebleu
        has_sb = True
    except ImportError:
        has_sb = False

    pairs = []
    for line in test_lines:
        line = line.strip()
        if not line:
            continue
        ids = sp.EncodeAsIds(line)
        if len(ids) < prefix_tok + gen_tok:
            continue
        pairs.append((ids[:prefix_tok], ids[prefix_tok:prefix_tok + gen_tok]))
        if len(pairs) >= n_samples:
            break

    if not pairs:
        return {}

    results = {}
    for temp in [0.0, 0.5, 1.0, 1.5]:
        lbl = "greedy" if temp == 0 else f"t={temp}"
        hyps, refs, gen_ids_all = [], [], []
        for prompt, ref_ids in pairs:
            gen_ids = generate_temp(model, prompt, gen_tok, temp, device, cfg)
            hyps.append(sp.DecodeIds(gen_ids))
            refs.append(sp.DecodeIds(ref_ids))
            gen_ids_all.append(gen_ids)

        bleu = chrf = None
        if has_sb:
            bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
            chrf = sacrebleu.corpus_chrf(hyps, [refs]).score

        rouge_l = sum(compute_rouge_l(sp.EncodeAsIds(h), sp.EncodeAsIds(r))
                      for h, r in zip(hyps, refs)) / len(hyps)
        results[lbl] = {
            "bleu4":    bleu,
            "chrf":     chrf,
            "rouge_l":  rouge_l,
            "rep_rate": repetition_rate(gen_ids_all),
            "distinct1": distinct_n(gen_ids_all, 1),
            "distinct2": distinct_n(gen_ids_all, 2),
            "samples":  list(zip(
                [sp.DecodeIds(p) for p, _ in pairs[:3]],
                hyps[:3], refs[:3],
            )),
        }
        print(f"  [{lbl}] BLEU={bleu or 'N/A':.2f}  chrF={chrf or 'N/A':.2f}  "
              f"ROUGE-L={rouge_l:.4f}  rep={results[lbl]['rep_rate']:.3f}  "
              f"D1={results[lbl]['distinct1']:.3f}  D2={results[lbl]['distinct2']:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Training curves comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(log_normal, log_nope, out_path):
    """Side-by-side val loss + PPL comparison between standard and NoPE models."""
    def load_log(path):
        rows = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        rows.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
        return rows

    rows_n  = load_log(log_normal)
    rows_np = load_log(log_nope)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Ablation: Model H (with PE) vs Model H (no PE) — Training Curves",
                 fontsize=13, fontweight="bold")

    for ax, key, ylabel, title in [
        (axes[0], "val_loss", "Validation cross-entropy loss", "Val Loss"),
        (axes[1], "val_ppl",  "Validation perplexity (PPL)",   "Val PPL"),
    ]:
        if rows_n:
            steps_n = [r["step"] for r in rows_n if key in r]
            vals_n  = [r[key]    for r in rows_n if key in r]
            ax.plot(steps_n, vals_n, color="#1e50a0", lw=2, label="Model H (with PE)")
        if rows_np:
            steps_np = [r["step"] for r in rows_np if key in r]
            vals_np  = [r[key]    for r in rows_np if key in r]
            ax.plot(steps_np, vals_np, color="#c03000", lw=2,
                    linestyle="--", label="Model H (no PE)")
        ax.set_xlabel("Training step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


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
    print(f"[device] {device}\n")

    from dataset  import make_dataloader
    from model    import DecoderLM
    from config_H import CONFIG_H

    from model_nope      import DecoderLM_NoPE
    from config_ablation import ABLATION_CONFIG

    tok_path = os.path.join(BASE_DIR, CONFIG_H.tokenizer_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)

    data_root = os.path.join(BASE_DIR, "data", "hindi", "splits")
    report_dir = os.path.join(BASE_DIR, "report")
    os.makedirs(report_dir, exist_ok=True)

    def load_model(ModelClass, cfg, ckpt_rel):
        m = ModelClass(cfg).to(device)
        ckpt_path = os.path.join(BASE_DIR, ckpt_rel)
        if not os.path.exists(ckpt_path):
            print(f"  [warn] checkpoint not found: {ckpt_path}")
            return None
        ck = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(ck["model"])
        total, _ = m.count_parameters()
        print(f"  loaded {cfg.model_name}  step={ck.get('step','?')}  "
              f"val_loss={ck.get('val_loss','?'):.4f}  params={total:,}")
        return m

    results = {}

    for label, ModelClass, cfg, ckpt_rel in [
        ("with_pe",  DecoderLM,     CONFIG_H,       "phase2/checkpoints_H/best.pt"),
        ("no_pe",    DecoderLM_NoPE, ABLATION_CONFIG, "phase2/ablation/checkpoints/best.pt"),
    ]:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {label}")
        print(f"{'='*60}")

        model = load_model(ModelClass, cfg, ckpt_rel)
        if model is None:
            results[label] = {"error": "checkpoint not found"}
            continue

        # PPL / BPB
        ppl_res = {}
        for split in ["val", "test"]:
            fpath = os.path.join(data_root, f"{split}.txt")
            if not os.path.exists(fpath):
                continue
            loader = make_dataloader(fpath, tok_path, cfg.context_len,
                                     cfg.batch_size, shuffle=False)
            loss, ppl, bpb = eval_ppl_bpb(model, loader, device, sp)
            print(f"  {split:<5}: loss={loss:.4f}  PPL={ppl:.2f}  BPB={bpb:.4f}")
            ppl_res[split] = {"loss": round(loss, 4), "ppl": round(ppl, 2),
                               "bpb": round(bpb, 4)}

        # Generation
        print("\n  --- Generation Quality ---")
        test_path = os.path.join(data_root, "test.txt")
        gen_res = {}
        if os.path.exists(test_path):
            with open(test_path, encoding="utf-8") as f:
                test_lines = f.readlines()
            gen_res = eval_generation(model, sp, test_lines, cfg, device)

        results[label] = {
            "model":      cfg.model_name,
            "ppl_bpb":    ppl_res,
            "generation": gen_res,
        }

        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    # Training curves
    plot_training_curves(
        os.path.join(BASE_DIR, "phase2/logs/train_H.jsonl"),
        os.path.join(BASE_DIR, "phase2/ablation/logs/train_nope.jsonl"),
        os.path.join(report_dir, "ablation_curves.png"),
    )

    # Comparison summary table
    print(f"\n{'='*60}")
    print("  ABLATION COMPARISON SUMMARY")
    print(f"{'='*60}")
    header = f"{'Metric':<30} {'With PE':>12} {'No PE':>12} {'Δ (NoPE − PE)':>14}"
    print(header)
    print("-" * len(header))
    for split in ["val", "test"]:
        for metric in ["loss", "ppl", "bpb"]:
            v_pe  = results.get("with_pe", {}).get("ppl_bpb", {}).get(split, {}).get(metric)
            v_npe = results.get("no_pe",   {}).get("ppl_bpb", {}).get(split, {}).get(metric)
            if v_pe is None or v_npe is None:
                continue
            delta = v_npe - v_pe
            print(f"  {split} {metric:<26} {v_pe:>12.4f} {v_npe:>12.4f} {delta:>+14.4f}")

    for temp_lbl in ["greedy", "t=1.0"]:
        for metric in ["bleu4", "rouge_l", "distinct1", "distinct2", "rep_rate"]:
            v_pe  = (results.get("with_pe", {}).get("generation", {})
                            .get(temp_lbl, {}).get(metric))
            v_npe = (results.get("no_pe",   {}).get("generation", {})
                            .get(temp_lbl, {}).get(metric))
            if v_pe is None or v_npe is None:
                continue
            delta = (v_npe - v_pe) if isinstance(v_npe, float) else "N/A"
            label_str = f"{temp_lbl} {metric}"
            print(f"  {label_str:<28} {v_pe:>12.4f} {v_npe:>12.4f} "
                  f"{delta:>+14.4f}" if isinstance(delta, float) else
                  f"  {label_str:<28} {v_pe:>12.4f} {v_npe:>12.4f} {'N/A':>14}")

    # Save JSON
    out_path = os.path.join(report_dir, "ablation_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
