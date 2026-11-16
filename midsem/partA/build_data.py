"""
A0 + A3: build the extra test sets and the DPO preference pairs.

Phase 3 shipped one test set (phase3/data/hindi_reasoning/test.jsonl, 6 held-out
names = UNSEEN ENTITIES).  The brief asks for three, so we add:

  test_seen_entities.jsonl  – 200 fresh questions generated with the *training*
                              templates and *training* names (same generator,
                              new seed), with every input that also occurs in
                              train.jsonl removed.
  test_unseen_wording.jsonl – 200 questions with *training* names but with
                              context sentences and question phrasings that
                              never occur in train.jsonl (checked below).  The
                              answer format is unchanged, so only the wording
                              of the input differs.

Preference pairs (A3) are built only from train.jsonl:
  chosen   = gold answer
  rejected = a plausible wrong answer (the other entity in a 2-way comparison,
             a wrong entity in a 3-way question, "X is higher" for an equal-value
             question, a wrongly ordered list for sorting questions).

Run from lm_project/:  python midsem/partA/build_data.py
"""
import os, sys, re, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import BASE, MID, DATA_DIR, read_jsonl, write_jsonl

sys.path.insert(0, os.path.join(BASE, "phase3"))
import generate_hindi_reasoning as G

N_TEST, N_PAIRS = 200, 2000
OUT = os.path.join(MID, "data")


# ── seen entities ────────────────────────────────────────────────────────────

def build_seen(train_inputs):
    rng, rows = random.Random(1234), []
    while len(rows) < N_TEST:
        fn = rng.choices(G.TEMPLATES, weights=G.WEIGHTS, k=1)[0]
        ex = fn(G.TRAIN_NAMES, rng)
        if ex["input"] not in train_inputs:
            rows.append(ex)
    return rows


# ── unseen wording (new phrasings, train names) ──────────────────────────────

NEW_Q_MORE_2 = ["{n1} या {n2}, इनमें से किसकी {prop} अधिक है?",
                "इन दोनों में से अधिक {prop} किसकी है?"]
NEW_Q_LESS_2 = ["{n1} या {n2}, इनमें से किसकी {prop} कम है?",
                "इन दोनों में से कम {prop} किसकी है?"]
NEW_Q_MAX_3  = ["इन तीनों में से सबसे अधिक {prop} वाला कौन है?",
                "{prop} के मामले में सबसे ऊपर कौन है?"]
NEW_Q_MIN_3  = ["इन तीनों में से सबसे कम {prop} वाला कौन है?",
                "{prop} के मामले में सबसे नीचे कौन है?"]
NEW_PHRASES = NEW_Q_MORE_2 + NEW_Q_LESS_2 + NEW_Q_MAX_3 + NEW_Q_MIN_3 + [
    "के हिसाब से छोटे से बड़े क्रम में लिखिए", "के हिसाब से बड़े से छोटे क्रम में लिखिए",
    "क्या किसी की {prop} दूसरे से अधिक है?", "इनमें से किसके पास ज़्यादा", "इनमें से किसके पास कम"]


def w1(rng):
    n1, n2 = rng.sample(G.TRAIN_NAMES, 2); prop, unit, (lo, hi) = rng.choice(G.PROPS)
    v1, v2 = G.distinct_vals(lo, hi, 2, rng)
    ctx = f"{n1} की {prop} {G.vs(v1, unit)} है और {n2} की {prop} {G.vs(v2, unit)} है।"
    if rng.random() < 0.5:
        q, a = rng.choice(NEW_Q_MORE_2), f"{n1 if v1 > v2 else n2} की {prop} अधिक है।"
    else:
        q, a = rng.choice(NEW_Q_LESS_2), f"{n1 if v1 < v2 else n2} की {prop} कम है।"
    return {"input": f"{ctx} {q.format(n1=n1, n2=n2, prop=prop)}", "output": a, "template": "T1", "property": prop}


def w2(rng):
    n1, n2 = rng.sample(G.TRAIN_NAMES, 2); prop, _, _ = rng.choice(G.PROPS)
    if rng.random() < 0.5:   # n1 > n2, phrased as "n2 is less than n1"
        ctx = f"{n2} की {prop} {n1} की तुलना में कम है।"
        q, a = rng.choice(NEW_Q_MORE_2), f"{n1} की {prop} अधिक है।"
    else:
        ctx = f"{n2} की {prop} {n1} की तुलना में अधिक है।"
        q, a = rng.choice(NEW_Q_LESS_2), f"{n1} की {prop} कम है।"
    return {"input": f"{ctx} {q.format(n1=n1, n2=n2, prop=prop)}", "output": a, "template": "T2", "property": prop}


def w3(rng):
    ns = rng.sample(G.TRAIN_NAMES, 3); prop, unit, (lo, hi) = rng.choice(G.PROPS)
    vs = G.distinct_vals(lo, hi, 3, rng)
    ctx = (f"{ns[0]} की {prop} {G.vs(vs[0], unit)}, {ns[1]} की {G.vs(vs[1], unit)} "
           f"तथा {ns[2]} की {G.vs(vs[2], unit)} है।")
    if rng.random() < 0.5:
        q, a = rng.choice(NEW_Q_MAX_3), f"{ns[vs.index(max(vs))]} की {prop} सबसे अधिक है।"
    else:
        q, a = rng.choice(NEW_Q_MIN_3), f"{ns[vs.index(min(vs))]} की {prop} सबसे कम है।"
    return {"input": f"{ctx} {q.format(prop=prop)}", "output": a, "template": "T3", "property": prop}


def w4(rng):   # chain n1 > n2 > n3, stated with "कम" in reverse order
    n1, n2, n3 = rng.sample(G.TRAIN_NAMES, 3); prop, _, _ = rng.choice(G.PROPS)
    ctx = f"{n3} की {prop} {n2} की तुलना में कम है, और {n2} की {prop} {n1} की तुलना में कम है।"
    if rng.random() < 0.5:
        q, a = rng.choice(NEW_Q_MAX_3), f"{n1} की {prop} सबसे अधिक है।"
    else:
        q, a = rng.choice(NEW_Q_MIN_3), f"{n3} की {prop} सबसे कम है।"
    return {"input": f"{ctx} {q.format(prop=prop)}", "output": a, "template": "T4", "property": prop}


def w6(rng):
    ns = rng.sample(G.TRAIN_NAMES, 3); prop, unit, (lo, hi) = rng.choice(G.PROPS)
    vs = G.distinct_vals(lo, hi, 3, rng)
    ctx = (f"{ns[0]} की {prop} {G.vs(vs[0], unit)}, {ns[1]} की {G.vs(vs[1], unit)} "
           f"तथा {ns[2]} की {G.vs(vs[2], unit)} है।")
    o = [n for _, n in sorted(zip(vs, ns))]
    if rng.random() < 0.5:
        q, a = f"{prop} के हिसाब से छोटे से बड़े क्रम में लिखिए।", f"{o[0]} < {o[1]} < {o[2]}"
    else:
        q, a = f"{prop} के हिसाब से बड़े से छोटे क्रम में लिखिए।", f"{o[2]} > {o[1]} > {o[0]}"
    return {"input": f"{ctx} {q}", "output": a, "template": "T6", "property": prop}


def w7(rng):
    n1, n2 = rng.sample(G.TRAIN_NAMES, 2); prop, unit, (lo, hi) = rng.choice(G.PROPS)
    v = rng.randint(lo, hi)
    ctx = f"{n1} की {prop} {G.vs(v, unit)} है और {n2} की {prop} भी {G.vs(v, unit)} है।"
    q = f"क्या किसी की {prop} दूसरे से अधिक है?"
    return {"input": f"{ctx} {q}", "output": f"दोनों की {prop} बराबर है।", "template": "T7", "property": prop}


def w8(rng):
    n1, n2 = rng.sample(G.TRAIN_NAMES, 2); item = rng.choice(G.ITEMS)
    v1 = rng.randint(1, 50); v2 = rng.randint(1, 50)
    while v2 == v1:
        v2 = rng.randint(1, 50)
    ctx = f"{n1} के पास {v1} {item} और {n2} के पास {v2} {item} हैं।"
    if rng.random() < 0.5:
        q, a = f"इनमें से किसके पास ज़्यादा {item} हैं?", f"{n1 if v1 > v2 else n2} के पास अधिक {item} हैं।"
    else:
        q, a = f"इनमें से किसके पास कम {item} हैं?", f"{n1 if v1 < v2 else n2} के पास कम {item} हैं।"
    return {"input": f"{ctx} {q}", "output": a, "template": "T8", "property": item}


def build_unseen_wording(train_inputs):
    rng, rows = random.Random(5678), []
    fns, wts = [w1, w2, w3, w4, w6, w7, w8], [2, 1, 2, 2, 1, 1, 1]
    while len(rows) < N_TEST:
        ex = rng.choices(fns, weights=wts, k=1)[0](rng)
        if ex["input"] not in train_inputs:
            rows.append(ex)
    return rows


# ── preference pairs ─────────────────────────────────────────────────────────

def names_in(text, pool):
    return [n for n in pool if re.search(rf"(^|[\s,]){re.escape(n)}(?=[\s,])", text)]


def make_rejected(ex, rng):
    out, t, prop = ex["output"], ex["template"], ex["property"]
    ents = names_in(ex["input"], G.TRAIN_NAMES)
    if t == "T6":                                   # wrong order: swap two neighbours
        sep = " < " if " < " in out else " > "
        o = out.split(sep); i = rng.randint(0, 1); o[i], o[i + 1] = o[i + 1], o[i]
        return sep.join(o), "swapped_order"
    if t == "T7":                                   # equal values → claim one is higher
        return f"{rng.choice(ents)} की {prop} अधिक है।", "false_inequality"
    gold = out.split()[0]
    wrong = rng.choice([n for n in ents if n != gold])
    return wrong + out[len(gold):], "other_entity"


def build_pairs(train):
    rng = random.Random(99)
    idx = rng.sample(range(len(train)), N_PAIRS)
    pairs = []
    for i in idx:
        ex = train[i]
        rej, kind = make_rejected(ex, rng)
        assert rej != ex["output"]
        pairs.append({"prompt": ex["input"], "chosen": ex["output"], "rejected": rej,
                      "rejected_type": kind, "template": ex["template"], "train_index": i})
    return pairs


def main():
    train = read_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    train_inputs = {e["input"] for e in train}
    train_text = "\n".join(e["input"] for e in train)

    seen = build_seen(train_inputs)
    wording = build_unseen_wording(train_inputs)
    pairs = build_pairs(train)

    # ── leakage checks ──
    for row in wording:   # no new-phrasing question/context fragment occurs in train
        for frag in ["इनमें से", "इन दोनों में से", "इन तीनों में से", "के मामले में",
                     "के हिसाब से", "की तुलना में", "दूसरे से अधिक", " तथा ", " भी "]:
            assert frag not in train_text, frag
    test_unseen = read_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
    test_inputs = {e["input"] for e in seen + wording + test_unseen}
    for p in pairs:
        assert not names_in(p["prompt"] + " " + p["chosen"] + " " + p["rejected"], G.TEST_NAMES + G.VAL_NAMES)
        assert p["prompt"] not in test_inputs
        for frag in ["इनमें से", "के मामले में", "के हिसाब से", "की तुलना में"]:
            assert frag not in p["prompt"]
    assert not (set(e["input"] for e in seen) & train_inputs)

    write_jsonl(os.path.join(OUT, "test_seen_entities.jsonl"), seen)
    write_jsonl(os.path.join(OUT, "test_unseen_wording.jsonl"), wording)
    write_jsonl(os.path.join(OUT, "preference_pairs.jsonl"), pairs)
    from collections import Counter
    print("seen:", len(seen), "unseen_wording:", len(wording), "pairs:", len(pairs))
    print("rejected types:", Counter(p["rejected_type"] for p in pairs))
    print("leakage checks passed")


if __name__ == "__main__":
    main()
