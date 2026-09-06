"""
Phase 2 Evaluation — Intrinsic LM metrics + Generation quality.

Computes for both Model H and Model L:
  1. Val/test cross-entropy loss, PPL, and BPB
  2. Greedy + temperature (0.5, 1.0, 1.5) generation from held-out prefixes
  3. BLEU-4, chrF, ROUGE-L against reference continuations
  4. Distinct-1/2 and repetition rate (diversity diagnostics)
  5. Qualitative generation samples

Usage:
    pip install sacrebleu rouge-score
    python phase2/eval_metrics.py --model H
    python phase2/eval_metrics.py --model L
    python phase2/eval_metrics.py --model H --model L   # both side-by-side
"""

import os, sys, math, argparse, json, re
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_bpb(loss_nats, tokenizer, sample_token_ids):
    """
    BPB = cross-entropy (nats/token) / (ln2 * avg_utf8_bytes_per_token).
    Normalises perplexity by byte length so models with different tokenizer
    fertility become comparable.
    """
    total_bytes  = sum(len(tokenizer.IdToPiece(tid).encode("utf-8"))
                       for seq in sample_token_ids for tid in seq)
    total_tokens = sum(len(seq) for seq in sample_token_ids)
    avg_bytes    = total_bytes / max(total_tokens, 1)
    return loss_nats / (math.log(2) * avg_bytes)


def repetition_rate(token_ids_list, n=4):
    """Fraction of n-grams in generated output that are repeated."""
    total, repeated = 0, 0
    for ids in token_ids_list:
        ngrams = [tuple(ids[i:i+n]) for i in range(len(ids) - n + 1)]
        total   += len(ngrams)
        counts   = Counter(ngrams)
        repeated += sum(v - 1 for v in counts.values())
    return repeated / max(total, 1)


def distinct_n(token_ids_list, n):
    """Distinct-n: unique n-grams / total n-grams across all generated sequences."""
    all_ngrams, unique = [], set()
    for ids in token_ids_list:
        for i in range(len(ids) - n + 1):
            g = tuple(ids[i:i+n])
            all_ngrams.append(g)
            unique.add(g)
    return len(unique) / max(len(all_ngrams), 1)


def compute_rouge_l(hyp_tokens, ref_tokens):
    """ROUGE-L F1 via longest common subsequence."""
    m, n = len(ref_tokens), len(hyp_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if ref_tokens[i-1] == hyp_tokens[j-1] \
                       else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / n
    r = lcs / m
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PPL / BPB evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_ppl_bpb(model, loader, device, tokenizer, max_batches=200):
    """Return (avg_loss, PPL, BPB) over up to max_batches batches."""
    model.eval()
    total_loss, n = 0.0, 0
    sample_ids = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
        if i < 10:  # collect a sample for BPB byte estimation
            sample_ids.extend(x.cpu().tolist())
    avg_loss = total_loss / max(n, 1)
    ppl      = math.exp(avg_loss)
    bpb      = compute_bpb(avg_loss, tokenizer, sample_ids)
    return avg_loss, ppl, bpb


# ─────────────────────────────────────────────────────────────────────────────
# Generation evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_at_temperature(model, prompt_ids, max_new, temperature, device, cfg):
    """Generate max_new tokens from prompt_ids at a given temperature."""
    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        idx_cond = idx[:, -cfg.context_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if temperature == 0:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs  = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
    return idx[0, len(prompt_ids):].tolist()  # return only the new tokens


def eval_generation(model, tokenizer, test_lines, cfg, device,
                    n_samples=100, prefix_tokens=64, gen_tokens=50):
    """
    Split each test line into prefix (first prefix_tokens tokens) + reference
    (next gen_tokens tokens).  Generate at three temperatures and score.
    Returns a dict of results per temperature.
    """
    try:
        import sacrebleu
        has_sacrebleu = True
    except ImportError:
        has_sacrebleu = False
        print("  [warn] sacrebleu not found — BLEU and chrF skipped. "
              "pip install sacrebleu")

    results = {}
    temperatures = [0.0, 0.5, 1.0, 1.5]  # 0.0 = greedy

    # Build prefix/reference pairs from test lines
    pairs = []
    for line in test_lines:
        line = line.strip()
        if not line:
            continue
        ids = tokenizer.EncodeAsIds(line)
        if len(ids) < prefix_tokens + gen_tokens:
            continue
        pairs.append((ids[:prefix_tokens], ids[prefix_tokens:prefix_tokens + gen_tokens]))
        if len(pairs) >= n_samples:
            break

    if not pairs:
        print("  [warn] No test lines long enough for generation evaluation.")
        return {}

    for temp in temperatures:
        label = "greedy" if temp == 0.0 else f"t={temp}"
        hyps, refs = [], []
        gen_ids_all = []

        for prompt_ids, ref_ids in pairs:
            gen_ids = generate_at_temperature(
                model, prompt_ids, gen_tokens, temp, device, cfg)
            hyp_text = tokenizer.DecodeIds(gen_ids)
            ref_text = tokenizer.DecodeIds(ref_ids)
            hyps.append(hyp_text)
            refs.append(ref_text)
            gen_ids_all.append(gen_ids)

        # BLEU-4 and chrF
        bleu_score, chrf_score = None, None
        if has_sacrebleu:
            bleu  = sacrebleu.corpus_bleu(hyps, [refs])
            chrf  = sacrebleu.corpus_chrf(hyps, [refs])
            bleu_score = bleu.score
            chrf_score = chrf.score

        # ROUGE-L (macro-average over samples)
        rouge_scores = []
        for hyp_text, ref_text in zip(hyps, refs):
            h_ids = tokenizer.EncodeAsIds(hyp_text)
            r_ids = tokenizer.EncodeAsIds(ref_text)
            rouge_scores.append(compute_rouge_l(h_ids, r_ids))
        rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

        # Diversity
        rep_rate  = repetition_rate(gen_ids_all, n=4)
        dist1     = distinct_n(gen_ids_all, 1)
        dist2     = distinct_n(gen_ids_all, 2)

        results[label] = {
            "bleu4":    bleu_score,
            "chrf":     chrf_score,
            "rouge_l":  rouge_l,
            "rep_rate": rep_rate,
            "distinct1": dist1,
            "distinct2": dist2,
            "samples":  list(zip(
                [tokenizer.DecodeIds(p) for p, _ in pairs[:3]],
                hyps[:3],
                refs[:3],
            )),
        }
        bleu_str = f"{bleu_score:.2f}" if bleu_score is not None else "N/A"
        chrf_str = f"{chrf_score:.2f}" if chrf_score is not None else "N/A"
        print(f"  [{label}] BLEU-4={bleu_str} | chrF={chrf_str} | "
              f"ROUGE-L={rouge_l:.4f} | rep={rep_rate:.3f} | "
              f"D1={dist1:.3f} | D2={dist2:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(cfg, base_dir, output_json):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    from model import DecoderLM
    from dataset import make_dataloader
    import sentencepiece as spm

    # Load tokenizer
    tok_path = os.path.join(base_dir, cfg.tokenizer_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)

    # Load model
    model = DecoderLM(cfg).to(device)
    ckpt_path = os.path.join(base_dir, cfg.checkpoint_dir, "best.pt")
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"[checkpoint] loaded step={ck.get('step', '?')}, "
          f"val_loss={ck.get('val_loss', '?'):.4f}\n")

    total, trainable = model.count_parameters()
    print(f"[params] total={total:,}  trainable={trainable:,}\n")

    data_root = os.path.join(base_dir, "data", cfg.language, "splits")

    # ── PPL / BPB ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Perplexity / BPB")
    print("=" * 60)
    ppl_results = {}
    for split in ["val", "test"]:
        fpath = os.path.join(data_root, f"{split}.txt")
        if not os.path.exists(fpath):
            print(f"  {split}: not found, skipping")
            continue
        loader = make_dataloader(fpath, tok_path, cfg.context_len,
                                 cfg.batch_size, shuffle=False)
        loss, ppl, bpb = eval_ppl_bpb(model, loader, device, sp)
        print(f"  {split:<5}: loss={loss:.4f}  PPL={ppl:.2f}  BPB={bpb:.4f}")
        ppl_results[split] = {"loss": loss, "ppl": ppl, "bpb": bpb}

    # ── Generation quality ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generation Quality")
    print("=" * 60)
    test_path = os.path.join(data_root, "test.txt")
    gen_results = {}
    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            test_lines = f.readlines()
        gen_results = eval_generation(
            model, sp, test_lines, cfg, device,
            n_samples=50, prefix_tokens=64, gen_tokens=50,
        )
    else:
        print("  test.txt not found — skipping generation evaluation")

    # ── Save results ─────────────────────────────────────────────────────────
    out = {
        "model": cfg.model_name,
        "language": cfg.language,
        "params_total": total,
        "params_trainable": trainable,
        "ppl_bpb": ppl_results,
        "generation": {k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                       for k, v in gen_results.items()},
        "generation_samples": {k: v.get("samples", [])
                                for k, v in gen_results.items()},
    }
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {output_json}")
    return out


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["H", "L"], action="append",
                        dest="models", default=None)
    args = parser.parse_args()
    if not args.models:
        args.models = ["H", "L"]

    base_dir = os.path.join(os.path.dirname(__file__), "..")

    all_results = {}
    for m in args.models:
        print(f"\n{'#'*60}")
        print(f"# Model {m}")
        print(f"{'#'*60}")
        if m == "H":
            from config_H import CONFIG_H as cfg
        else:
            from config_L import CONFIG_L as cfg

        out_json = os.path.join(base_dir, "report", f"eval_model_{m}.json")
        result = run_evaluation(cfg, base_dir, out_json)
        all_results[m] = result

    # ── Side-by-side comparison table ────────────────────────────────────────
    if len(all_results) == 2:
        print("\n" + "=" * 70)
        print("SIDE-BY-SIDE COMPARISON")
        print("=" * 70)
        h, l = all_results.get("H", {}), all_results.get("L", {})
        print(f"{'Metric':<25} {'Model H':>12} {'Model L':>12}")
        print("-" * 50)
        for split in ["val", "test"]:
            for metric in ["loss", "ppl", "bpb"]:
                hv = h.get("ppl_bpb", {}).get(split, {}).get(metric, "—")
                lv = l.get("ppl_bpb", {}).get(split, {}).get(metric, "—")
                label = f"{split}_{metric}"
                hstr = f"{hv:.4f}" if isinstance(hv, float) else str(hv)
                lstr = f"{lv:.4f}" if isinstance(lv, float) else str(lv)
                print(f"  {label:<23} {hstr:>12} {lstr:>12}")
        print()
        for temp_key in ["greedy", "t=0.5", "t=1.0", "t=1.5"]:
            for metric in ["bleu4", "chrf", "rouge_l", "distinct1", "distinct2", "rep_rate"]:
                hv = h.get("generation", {}).get(temp_key, {}).get(metric, "—")
                lv = l.get("generation", {}).get(temp_key, {}).get(metric, "—")
                label = f"{temp_key}_{metric}"
                hstr = f"{hv:.4f}" if isinstance(hv, float) else str(hv)
                lstr = f"{lv:.4f}" if isinstance(lv, float) else str(lv)
                print(f"  {label:<23} {hstr:>12} {lstr:>12}")
