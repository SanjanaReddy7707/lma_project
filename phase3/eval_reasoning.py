"""
Reasoning evaluation — pretrained vs. fine-tuned models on the test set.

Metrics
-------
* Exact-match (EM): normalized predicted answer == normalized gold answer
* Entity accuracy: first token of gold answer appears in prediction
  (lenient; catches cases where wording differs but entity is correct)

Outputs
-------
  report/eval_reasoning_H.json
  report/eval_reasoning_L.json

Each JSON contains:
  {
    "model": "Model_H",
    "pretrained": { "em_acc": ..., "ent_acc": ..., "samples": [...] },
    "finetuned":  { "em_acc": ..., "ent_acc": ..., "samples": [...] },
    "improvement": { "em": ..., "ent": ... },
    "by_template": { "T1": {...}, ... }
  }

Usage
-----
    python phase3/eval_reasoning.py --model H
    python phase3/eval_reasoning.py --model L
    python phase3/eval_reasoning.py --model H --model L
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import sentencepiece as spm

PROMPT_PREFIX = "प्रश्न: "
ANSWER_PREFIX = "\nउत्तर: "


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(text):
    return text.strip().rstrip("।\n").strip()


@torch.no_grad()
def greedy_generate(model, prompt_ids, max_new, eos_id, device, cfg):
    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    for _ in range(max_new):
        idx_cond  = idx[:, -cfg.context_len:]
        logits, _ = model(idx_cond)
        next_id   = logits[0, -1].argmax().item()
        if next_id == eos_id:
            break
        generated.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    return generated


def load_model(ckpt_path, cfg, device):
    from phase2.model import DecoderLM
    model = DecoderLM(cfg).to(device)
    ck    = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    step  = ck.get("step", "?")
    print(f"  Loaded {ckpt_path} (step={step})")
    return model


def evaluate_model(model, test_examples, sp, cfg, device, n_samples=None):
    """
    Run greedy generation on test examples.
    Returns accuracy dict and per-example samples.
    """
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    em_correct  = 0
    ent_correct = 0
    by_template: dict[str, dict] = {}
    samples = []

    examples = test_examples if n_samples is None else test_examples[:n_samples]

    for ex in examples:
        q_text   = f"{PROMPT_PREFIX}{ex['input']}{ANSWER_PREFIX}"
        prompt   = [bos_id] + sp.EncodeAsIds(q_text)
        gold_ans = normalize(ex["output"])
        tmpl     = ex.get("template", "?")

        gen_ids  = greedy_generate(model, prompt, max_new=80,
                                   eos_id=eos_id, device=device, cfg=cfg)
        pred_ans = normalize(sp.DecodeIds(gen_ids))

        em  = (pred_ans == gold_ans)
        ent = bool(gold_ans.split()[0]) and gold_ans.split()[0] in pred_ans

        em_correct  += int(em)
        ent_correct += int(ent)

        # Per-template stats
        if tmpl not in by_template:
            by_template[tmpl] = {"em": 0, "ent": 0, "n": 0}
        by_template[tmpl]["em"]  += int(em)
        by_template[tmpl]["ent"] += int(ent)
        by_template[tmpl]["n"]   += 1

        if len(samples) < 10:
            samples.append({
                "input":  ex["input"],
                "gold":   gold_ans,
                "pred":   pred_ans,
                "em":     em,
                "ent":    ent,
                "template": tmpl,
            })

    n = len(examples)
    for tmpl, d in by_template.items():
        d["em_acc"]  = round(d["em"]  / max(d["n"], 1), 4)
        d["ent_acc"] = round(d["ent"] / max(d["n"], 1), 4)

    return {
        "em_acc":    round(em_correct  / max(n, 1), 4),
        "ent_acc":   round(ent_correct / max(n, 1), 4),
        "n":         n,
        "samples":   samples,
        "by_template": by_template,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(model_key, base_dir):
    if model_key == "H":
        from config_ft_H import CONFIG_FT_H as cfg
        pretrained_ckpt = os.path.join(base_dir, "phase2/checkpoints_H/best.pt")
        finetuned_ckpt  = os.path.join(base_dir, "phase3/checkpoints_ft_H/best_ft.pt")
    else:
        from config_ft_L import CONFIG_FT_L as cfg
        pretrained_ckpt = os.path.join(base_dir, "phase2/checkpoints_L/best.pt")
        finetuned_ckpt  = os.path.join(base_dir, "phase3/checkpoints_ft_L/best_ft.pt")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    tok_path = os.path.join(base_dir, cfg.tokenizer_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)

    test_path = os.path.join(base_dir, cfg.data_dir, "test.jsonl")
    test_examples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_examples.append(json.loads(line))
    print(f"[test] {len(test_examples)} examples")

    print(f"\n{'='*60}")
    print(f"  Pretrained Model {model_key}")
    print(f"{'='*60}")
    pretrained_model = load_model(pretrained_ckpt, cfg, device)
    pre_results = evaluate_model(pretrained_model, test_examples, sp, cfg, device)
    print(f"  EM={pre_results['em_acc']:.3f}  EntityAcc={pre_results['ent_acc']:.3f}")
    del pretrained_model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  Finetuned Model {model_key}")
    print(f"{'='*60}")
    ft_results = {}
    if os.path.exists(finetuned_ckpt):
        finetuned_model = load_model(finetuned_ckpt, cfg, device)
        ft_results = evaluate_model(finetuned_model, test_examples, sp, cfg, device)
        print(f"  EM={ft_results['em_acc']:.3f}  EntityAcc={ft_results['ent_acc']:.3f}")
        del finetuned_model
    else:
        print(f"  [warn] Finetuned checkpoint not found: {finetuned_ckpt}")
        ft_results = {"em_acc": None, "ent_acc": None, "n": 0, "samples": [],
                      "by_template": {}}

    improvement = {}
    if ft_results.get("em_acc") is not None:
        improvement = {
            "em":  round(ft_results["em_acc"]  - pre_results["em_acc"],  4),
            "ent": round(ft_results["ent_acc"] - pre_results["ent_acc"], 4),
        }

    out = {
        "model":       cfg.model_name,
        "language":    cfg.language,
        "n_test":      len(test_examples),
        "pretrained":  pre_results,
        "finetuned":   ft_results,
        "improvement": improvement,
    }

    out_path = os.path.join(base_dir, "report", f"eval_reasoning_{model_key}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out_path}")
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
