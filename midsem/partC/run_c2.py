"""
C2: Can you trust an LLM judge?

Pairs  : 30 problems-slots built from C1 outputs. Each pair = (direct answer, one
         chain-of-thought answer) for the same problem. CoT answers come from the
         greedy CoT run or the 5 self-consistency samples. Selection deliberately
         includes pairs where the SHORT (direct) answer is right and the LONG one
         is wrong.  Ground truth = checker.
Judge  : microsoft/Phi-3.5-mini-instruct (different family from Qwen), greedy,
         asked to output "A" or "B".  Every pair is judged twice (direct as A,
         then direct as B).
Metrics: position-bias rate (out of 30), agreement with checker on
         order-consistent verdicts, Cohen's kappa, longer-answer win rate (60).

Run: python midsem/partC/run_c2.py
"""
import os, sys, json, random, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import MID, read_jsonl, write_jsonl, save_json, get_device

JUDGE_ID = "microsoft/Phi-3.5-mini-instruct"
N_PAIRS = 30

JUDGE_PROMPT = """You are judging two answers to the same math word problem (written in Hindi).
Decide which answer is better, meaning which one reaches the correct final number.

Problem:
{q}

Answer A:
{a}

Answer B:
{b}

Which answer is better? Reply with exactly one letter: A or B."""


def build_pairs(rows, rng):
    """One candidate per (problem, CoT answer); prefer informative pairs."""
    cands = []
    for r in rows:
        d = r["direct"]
        cots = [("cot_greedy", r["cot"])] + [(f"sc_sample{i}", s) for i, s in enumerate(r["sc5"]["samples"])]
        for src, c in cots:
            if c["text"].strip() == d["text"].strip():
                continue
            cands.append({"problem_id": r["id"], "type": r["type"], "question": r["question"],
                          "gold": r["answer"],
                          "direct": {"text": d["text"], "correct": d["correct"], "n_tokens": d["n_tokens"]},
                          "cot": {"text": c["text"], "correct": c["correct"], "n_tokens": c["n_tokens"], "source": src}})
    def cat(p):
        dc, cc = p["direct"]["correct"], p["cot"]["correct"]
        return {(True, False): "short_right_long_wrong", (False, True): "short_wrong_long_right",
                (True, True): "both_right", (False, False): "both_wrong"}[(dc, cc)]
    for p in cands:
        p["category"] = cat(p)
    rng.shuffle(cands)
    # quotas: as many disagreement pairs as possible (they are the informative ones),
    # at most 2 pairs per problem, then fill with agreement pairs
    quota = {"short_right_long_wrong": 10, "short_wrong_long_right": 10, "both_right": 6, "both_wrong": 4}
    chosen, per_prob = [], {}
    for pass_ in range(2):
        for p in cands:
            if len(chosen) >= N_PAIRS:
                break
            if p in chosen or per_prob.get(p["problem_id"], 0) >= 2:
                continue
            if pass_ == 0 and sum(c["category"] == p["category"] for c in chosen) >= quota[p["category"]]:
                continue
            chosen.append(p); per_prob[p["problem_id"]] = per_prob.get(p["problem_id"], 0) + 1
    if len(chosen) < N_PAIRS:      # relax per-problem cap if still short
        for p in cands:
            if len(chosen) >= N_PAIRS:
                break
            if p not in chosen:
                chosen.append(p)
    for i, p in enumerate(chosen):
        p["pair_id"] = f"P{i+1:02d}"
    return chosen


@torch.no_grad()
def judge(model, tok, q, a, b, device):
    msgs = [{"role": "user", "content": JUDGE_PROMPT.format(q=q, a=a.strip(), b=b.strip())}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(s, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"\b([AB])\b", txt) or re.search(r"([AB])", txt)
    if m:
        return m.group(1), txt
    # fallback: compare next-token logits for "A" vs "B"
    logits = model(**enc).logits[0, -1]
    ia, ib = tok.convert_tokens_to_ids("A"), tok.convert_tokens_to_ids("B")
    return ("A" if logits[ia] >= logits[ib] else "B"), txt + " [logit-fallback]"


def cohen_kappa(y1, y2):
    n = len(y1)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(y1, y2)) / n
    labels = set(y1) | set(y2)
    pe = sum((y1.count(l) / n) * (y2.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    device = get_device()
    rng = random.Random(0)
    rows = read_jsonl(os.path.join(MID, "results/c1_generations.jsonl"))
    pairs = build_pairs(rows, rng)

    tok = AutoTokenizer.from_pretrained(JUDGE_ID)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_ID, dtype=torch.float16).to(device).eval()

    for p in pairs:
        # order 1: direct = A ; order 2: direct = B
        v1, raw1 = judge(model, tok, p["question"], p["direct"]["text"], p["cot"]["text"], device)
        v2, raw2 = judge(model, tok, p["question"], p["cot"]["text"], p["direct"]["text"], device)
        w1 = "direct" if v1 == "A" else "cot"
        w2 = "direct" if v2 == "B" else "cot"
        p["verdicts"] = [{"order": "direct=A,cot=B", "raw": raw1, "letter": v1, "winner": w1},
                         {"order": "cot=A,direct=B", "raw": raw2, "letter": v2, "winner": w2}]
        p["order_consistent"] = w1 == w2
        print(f"{p['pair_id']} {p['category']:<24} order1→{w1:<6} order2→{w2:<6}", flush=True)

    # ── metrics ──
    n = len(pairs)
    flips = sum(not p["order_consistent"] for p in pairs)
    # ground-truth label per pair: which answer is better according to the checker.
    # pairs where both are right / both wrong have no unique better answer → label "tie".
    def gt(p):
        dc, cc = p["direct"]["correct"], p["cot"]["correct"]
        return "direct" if dc and not cc else "cot" if cc and not dc else "tie"
    consistent = [p for p in pairs if p["order_consistent"]]
    decidable = [p for p in consistent if gt(p) != "tie"]
    agree = sum(p["verdicts"][0]["winner"] == gt(p) for p in decidable)
    kappa = cohen_kappa([p["verdicts"][0]["winner"] for p in decidable], [gt(p) for p in decidable])
    # longer answer = more generated tokens
    verdicts = [(p, v) for p in pairs for v in p["verdicts"]]
    def longer(p):
        return "cot" if p["cot"]["n_tokens"] >= p["direct"]["n_tokens"] else "direct"
    longer_wins = sum(v["winner"] == longer(p) for p, v in verdicts)
    letter_A = sum(v["letter"] == "A" for _, v in verdicts)
    srlw = [p for p in pairs if p["category"] == "short_right_long_wrong"]

    summ = {
        "judge": JUDGE_ID, "n_pairs": n,
        "category_counts": {c: sum(p["category"] == c for p in pairs) for c in
                            ["short_right_long_wrong", "short_wrong_long_right", "both_right", "both_wrong"]},
        "position_bias_rate": flips / n, "position_flips": flips,
        "order_consistent_pairs": len(consistent),
        "order_consistent_with_unique_gt": len(decidable),
        "agreement_with_checker": agree / len(decidable) if decidable else None,
        "agreement_count": agree,
        "cohens_kappa": kappa,
        "longer_answer_win_rate": longer_wins / len(verdicts), "longer_answer_wins": longer_wins,
        "n_verdicts": len(verdicts),
        "letter_A_rate": letter_A / len(verdicts),
        "short_right_long_wrong": {
            "n": len(srlw),
            "judge_picked_short_both_orders": sum(p["order_consistent"] and p["verdicts"][0]["winner"] == "direct" for p in srlw),
            "judge_picked_long_both_orders": sum(p["order_consistent"] and p["verdicts"][0]["winner"] == "cot" for p in srlw)},
    }
    write_jsonl(os.path.join(MID, "data/c2_pairs_with_verdicts.jsonl"), pairs)
    save_json(os.path.join(MID, "results/c2_summary.json"), summ)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
