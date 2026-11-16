"""
C1: Does thinking longer help?  Qwen2.5-1.5B-Instruct on 20 Hindi problems.

Conditions
  direct : greedy, prompt asks for only the final number
  cot    : greedy, zero-shot chain of thought ("चरण-दर-चरण सोचिए")
  sc5    : chain-of-thought prompt, 5 samples (T=0.7, top-p=0.95), majority vote
           over checker-normalised answers (ties → earliest sample)

Outputs: midsem/results/c1_generations.jsonl, midsem/results/c1_summary.json
Run: python midsem/partC/run_c1.py
"""
import os, sys, json, time
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import MID, read_jsonl, write_jsonl, save_json, get_device
from checker import extract_answer, is_correct, to_fraction

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SC_N, TEMP, TOP_P, SEED = 5, 0.7, 0.95, 0
MAX_NEW = {"direct": 32, "cot": 512}

SYS = "आप एक सहायक हैं जो गणित के शब्द-प्रश्न हल करते हैं।"
PROMPTS = {
    "direct": "{q}\nकेवल अंतिम उत्तर एक संख्या के रूप में दीजिए, कोई व्याख्या नहीं।",
    "cot":    "{q}\nचरण-दर-चरण सोचिए, फिर अंतिम पंक्ति में 'उत्तर: <संख्या>' लिखिए।",
}


def chat_ids(tok, text, device):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": text}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, return_tensors="pt").to(device)


@torch.no_grad()
def generate(model, tok, text, device, max_new, sample=False, n=1):
    enc = chat_ids(tok, text, device)
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    if sample:
        kw.update(do_sample=True, temperature=TEMP, top_p=TOP_P, num_return_sequences=n)
    else:
        kw.update(do_sample=False)
    out = model.generate(**enc, **kw)
    gen = out[:, enc["input_ids"].shape[1]:]
    res = []
    for row in gen:
        ids = row.tolist()
        if tok.eos_token_id in ids:          # count tokens up to and incl. EOS
            ids = ids[:ids.index(tok.eos_token_id) + 1]
        res.append({"text": tok.decode(ids, skip_special_tokens=True), "n_tokens": len(ids)})
    return res


def majority(answers):
    keys = [to_fraction(a) if a is not None else None for a in answers]
    valid = [k for k in keys if k is not None]
    if not valid:
        return None, 0
    cnt = Counter(valid)
    top = max(cnt.values())
    for a, k in zip(answers, keys):          # earliest sample among the tied winners
        if k is not None and cnt[k] == top:
            return a, top


def main():
    device = get_device()
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    probs = read_jsonl(os.path.join(MID, "data/c1_problems.jsonl"))

    rows, t0 = [], time.time()
    for p in probs:
        q = p["question"]
        d = generate(model, tok, PROMPTS["direct"].format(q=q), device, MAX_NEW["direct"])[0]
        c = generate(model, tok, PROMPTS["cot"].format(q=q), device, MAX_NEW["cot"])[0]
        torch.manual_seed(SEED + probs.index(p))   # reproducible per-problem seed
        s = generate(model, tok, PROMPTS["cot"].format(q=q), device, MAX_NEW["cot"], sample=True, n=SC_N)
        s_ans = [extract_answer(x["text"]) for x in s]
        vote, votes = majority(s_ans)
        row = {
            **p,
            "direct": {**d, "pred": extract_answer(d["text"]), "correct": is_correct(d["text"], p["answer"])},
            "cot":    {**c, "pred": extract_answer(c["text"]), "correct": is_correct(c["text"], p["answer"])},
            "sc5": {"samples": [{**x, "pred": a, "correct": is_correct(x["text"], p["answer"])}
                                for x, a in zip(s, s_ans)],
                    "vote": vote, "votes": votes,
                    "correct": vote is not None and is_correct(f"उत्तर: {vote}", p["answer"]),
                    "n_tokens": sum(x["n_tokens"] for x in s)},
        }
        rows.append(row)
        print(f"{p['id']} gold={p['answer']:>5} | direct={row['direct']['pred']} "
              f"cot={row['cot']['pred']} sc={vote}({votes}/5) | {time.time()-t0:.0f}s", flush=True)

    write_jsonl(os.path.join(MID, "results/c1_generations.jsonl"), rows)

    summ = {"model": MODEL_ID, "sc": {"n": SC_N, "temperature": TEMP, "top_p": TOP_P}, "conditions": {}}
    for cond in ["direct", "cot", "sc5"]:
        d = {}
        for typ in ["standard", "modified"]:
            sub = [r for r in rows if r["type"] == typ]
            d[f"acc_{typ}"] = sum(r[cond]["correct"] for r in sub) / len(sub)
            d[f"n_{typ}"] = len(sub)
        d["acc_all"] = sum(r[cond]["correct"] for r in rows) / len(rows)
        d["mean_gen_tokens"] = sum(r[cond]["n_tokens"] for r in rows) / len(rows)
        if cond == "sc5":
            d["mean_gen_tokens_per_sample"] = d["mean_gen_tokens"] / SC_N
        summ["conditions"][cond] = d
    # modified problems: how often did the model output the template (familiar) answer?
    mod = [r for r in rows if r["type"] == "modified"]
    summ["modified_template_answer_rate"] = {
        cond: sum(is_correct(f"उत्तर: {r[cond]['pred'] if cond != 'sc5' else r['sc5']['vote']}",
                             r["template_answer"]) for r in mod) / len(mod)
        for cond in ["direct", "cot", "sc5"]}
    # language check: share of CoT outputs containing Devanagari vs. switching to English/Chinese
    def script(t):
        dev = sum('ऀ' <= ch <= 'ॿ' for ch in t); lat = sum(ch.isascii() and ch.isalpha() for ch in t)
        han = sum('一' <= ch <= '鿿' for ch in t)
        return "hindi" if dev >= max(lat, han) else ("chinese" if han > lat else "english")
    summ["cot_output_language"] = dict(Counter(script(r["cot"]["text"]) for r in rows))
    save_json(os.path.join(MID, "results/c1_summary.json"), summ)
    print(json.dumps(summ, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
