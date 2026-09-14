"""
Synthetic Hindi reasoning dataset for fine-tuning Model H.

Templates cover:
  T1 — two-entity comparison with explicit numeric values
  T2 — two-entity relational comparison (no numbers, stated in words)
  T3 — three-entity comparison with explicit values (find max / min)
  T4 — three-entity chain  A > B > C  (find max / min / A-vs-C)
  T5 — three-entity chain  A < B < C  (find max / min)
  T6 — sort three entities ascending or descending
  T7 — equal-value case (two entities, same value)
  T8 — counting items (two entities, count comparison)

Entity names are split disjointly across train / val / test to prevent leakage.
Multiple question phrasings and property types create lexical variety.

Output
------
phase3/data/hindi_reasoning/{train,val,test}.jsonl

Each JSONL line
---------------
{"input": "<question>", "output": "<answer>", "template": "<T1..T8>", "property": "<prop>"}

Run from lm_project/
---------------------
python phase3/generate_hindi_reasoning.py
"""

import json
import random
import os

SEED = 42
random.seed(SEED)

OUT_DIR = "phase3/data/hindi_reasoning"

# ── Entity name sets (strictly disjoint to prevent split leakage) ─────────────
TRAIN_NAMES = [
    "राम", "श्याम", "सीता", "गीता", "मोहन", "सोहन",
    "अर्जुन", "नेहा", "प्रिया", "दीपक", "आनंद", "रमेश",
    "सुरेश", "महेश", "कमला", "कविता", "विक्रम", "भोला",
    "शोभा", "सुमन",
]
VAL_NAMES  = ["रोहित", "अनिता", "विकास", "पूजा", "संजय", "मीना"]
TEST_NAMES = ["अमित", "रेखा", "विनोद", "शांति", "कपिल", "लता"]

# ── Properties (name, unit, value-range) ─────────────────────────────────────
PROPS = [
    ("उम्र",     "साल",        (10, 80)),
    ("ऊँचाई",   "सेमी",       (140, 200)),
    ("गति",     "किमी/घंटा",  (20, 150)),
    ("कीमत",   "रुपये",       (100, 9999)),
    ("दूरी",    "किमी",        (1, 500)),
    ("तापमान", "डिग्री",      (5, 45)),
]

# ── Counting-item templates ───────────────────────────────────────────────────
ITEMS = ["आम", "किताबें", "संतरे", "पेंसिलें", "सेब"]

# ── Question phrasings ────────────────────────────────────────────────────────
Q_MORE_2 = [
    "किसकी {prop} अधिक है?",
    "किसकी {prop} ज़्यादा है?",
    "दोनों में से किसकी {prop} बड़ी है?",
    "{n1} और {n2} में किसकी {prop} अधिक है?",
]
Q_LESS_2 = [
    "किसकी {prop} कम है?",
    "दोनों में से किसकी {prop} छोटी है?",
    "{n1} और {n2} में किसकी {prop} कम है?",
    "दोनों में {prop} में कौन पीछे है?",
]
Q_MAX_3 = [
    "सबसे अधिक {prop} किसकी है?",
    "तीनों में {prop} में सबसे आगे कौन है?",
    "किसकी {prop} सबसे ज़्यादा है?",
]
Q_MIN_3 = [
    "सबसे कम {prop} किसकी है?",
    "तीनों में {prop} में सबसे पीछे कौन है?",
    "किसकी {prop} सबसे कम है?",
]


# ── Helper ────────────────────────────────────────────────────────────────────

def vs(v, unit):
    """Format a value with its unit (skip space when unit is empty)."""
    return f"{v} {unit}".strip() if unit else str(v)


def pk(lst, rng):
    return rng.choice(lst)


def distinct_vals(lo, hi, k, rng):
    """Sample k distinct integer values from [lo, hi]."""
    population = list(range(lo, hi + 1))
    rng.shuffle(population)
    return population[:k]


# ── Template generators ───────────────────────────────────────────────────────

def t1(names, rng):
    """Two entities, explicit values; ask more or less."""
    n1, n2 = rng.sample(names, 2)
    prop, unit, (lo, hi) = pk(PROPS, rng)
    v1, v2 = distinct_vals(lo, hi, 2, rng)

    ctx = f"{n1} की {prop} {vs(v1, unit)} है। {n2} की {prop} {vs(v2, unit)} है।"

    if rng.random() < 0.5:
        q   = pk(Q_MORE_2, rng).format(prop=prop, n1=n1, n2=n2)
        ans_name = n1 if v1 > v2 else n2
        ans = f"{ans_name} की {prop} अधिक है।"
    else:
        q   = pk(Q_LESS_2, rng).format(prop=prop, n1=n1, n2=n2)
        ans_name = n1 if v1 < v2 else n2
        ans = f"{ans_name} की {prop} कम है।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T1", "property": prop}


def t2(names, rng):
    """Two entities, relational comparison (no numbers)."""
    n1, n2 = rng.sample(names, 2)
    prop, unit, _ = pk(PROPS, rng)

    if rng.random() < 0.5:
        # n1 > n2
        ctx = f"{n1} की {prop} {n2} से अधिक है।"
        q   = f"{n1} और {n2} में किसकी {prop} कम है?"
        ans = f"{n2} की {prop} कम है।"
    else:
        # n1 < n2
        ctx = f"{n1} की {prop} {n2} से कम है।"
        q   = f"{n1} और {n2} में किसकी {prop} अधिक है?"
        ans = f"{n2} की {prop} अधिक है।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T2", "property": prop}


def t3(names, rng):
    """Three entities, explicit values; ask max or min."""
    n1, n2, n3 = rng.sample(names, 3)
    prop, unit, (lo, hi) = pk(PROPS, rng)
    v1, v2, v3 = distinct_vals(lo, hi, 3, rng)

    ctx = (f"{n1}, {n2} और {n3} की {prop} क्रमशः "
           f"{vs(v1, unit)}, {vs(v2, unit)} और {vs(v3, unit)} है।")

    vals = [v1, v2, v3]
    ns   = [n1, n2, n3]

    if rng.random() < 0.5:
        q   = pk(Q_MAX_3, rng).format(prop=prop)
        ans_name = ns[vals.index(max(vals))]
        ans = f"{ans_name} की {prop} सबसे अधिक है।"
    else:
        q   = pk(Q_MIN_3, rng).format(prop=prop)
        ans_name = ns[vals.index(min(vals))]
        ans = f"{ans_name} की {prop} सबसे कम है।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T3", "property": prop}


def t4(names, rng):
    """Chain: n1 > n2 > n3; ask max / min / n1-vs-n3."""
    n1, n2, n3 = rng.sample(names, 3)
    prop, unit, _ = pk(PROPS, rng)

    ctx = (f"{n1} की {prop} {n2} से अधिक है। "
           f"{n2} की {prop} {n3} से अधिक है।")

    choice = rng.randint(0, 2)
    if choice == 0:
        q   = pk(Q_MAX_3, rng).format(prop=prop)
        ans = f"{n1} की {prop} सबसे अधिक है।"
    elif choice == 1:
        q   = pk(Q_MIN_3, rng).format(prop=prop)
        ans = f"{n3} की {prop} सबसे कम है।"
    else:
        q   = f"{n1} और {n3} में किसकी {prop} कम है?"
        ans = f"{n3} की {prop} कम है।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T4", "property": prop}


def t5(names, rng):
    """Chain: n1 < n2 < n3; ask max / min."""
    n1, n2, n3 = rng.sample(names, 3)
    prop, unit, _ = pk(PROPS, rng)

    ctx = (f"{n1} की {prop} {n2} से कम है। "
           f"{n2} की {prop} {n3} से कम है।")

    if rng.random() < 0.5:
        q   = pk(Q_MAX_3, rng).format(prop=prop)
        ans = f"{n3} की {prop} सबसे अधिक है।"
    else:
        q   = pk(Q_MIN_3, rng).format(prop=prop)
        ans = f"{n1} की {prop} सबसे कम है।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T5", "property": prop}


def t6(names, rng):
    """Three entities with values; ask ascending or descending order."""
    n1, n2, n3 = rng.sample(names, 3)
    prop, unit, (lo, hi) = pk(PROPS, rng)
    v1, v2, v3 = distinct_vals(lo, hi, 3, rng)

    ctx = (f"{n1}, {n2} और {n3} की {prop} क्रमशः "
           f"{vs(v1, unit)}, {vs(v2, unit)} और {vs(v3, unit)} है।")

    pairs = sorted(zip([v1, v2, v3], [n1, n2, n3]))
    ordered = [p[1] for p in pairs]

    if rng.random() < 0.5:
        q   = f"{prop} के अनुसार बढ़ते क्रम में क्या है?"
        ans = f"{ordered[0]} < {ordered[1]} < {ordered[2]}"
    else:
        q   = f"{prop} के अनुसार घटते क्रम में क्या है?"
        ans = f"{ordered[2]} > {ordered[1]} > {ordered[0]}"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T6", "property": prop}


def t7(names, rng):
    """Two entities with equal values; ask who has more."""
    n1, n2 = rng.sample(names, 2)
    prop, unit, (lo, hi) = pk(PROPS, rng)
    v = rng.randint(lo, hi)

    ctx = f"{n1} और {n2} दोनों की {prop} {vs(v, unit)} है।"
    q   = f"दोनों में से किसकी {prop} अधिक है?"
    ans = f"दोनों की {prop} बराबर है।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T7", "property": prop}


def t8(names, rng):
    """Counting items: two entities with item counts; ask who has more."""
    n1, n2 = rng.sample(names, 2)
    item = pk(ITEMS, rng)
    v1 = rng.randint(1, 50)
    v2 = rng.randint(1, 50)
    while v2 == v1:
        v2 = rng.randint(1, 50)

    ctx = f"{n1} के पास {v1} {item} हैं। {n2} के पास {v2} {item} हैं।"

    if rng.random() < 0.5:
        q   = f"किसके पास अधिक {item} हैं?"
        ans_name = n1 if v1 > v2 else n2
        ans = f"{ans_name} के पास अधिक {item} हैं।"
    else:
        q   = f"किसके पास कम {item} हैं?"
        ans_name = n1 if v1 < v2 else n2
        ans = f"{ans_name} के पास कम {item} हैं।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T8", "property": item}


# ── Dataset builder ───────────────────────────────────────────────────────────

TEMPLATES = [t1, t2, t3, t4, t5, t6, t7, t8]
# Weights: T1 and T3 slightly more frequent (direct numeric comparison)
WEIGHTS    = [2, 1, 2, 2, 1, 1, 1, 1]


def generate_split(names, n, seed):
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        fn = rng.choices(TEMPLATES, weights=WEIGHTS, k=1)[0]
        ex = fn(names, rng)
        examples.append(ex)
    rng.shuffle(examples)
    return examples


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path}: {len(records)} examples")


if __name__ == "__main__":
    import collections
    print("Generating Hindi reasoning dataset …")

    train = generate_split(TRAIN_NAMES, 2000, seed=1)
    val   = generate_split(VAL_NAMES,   200,  seed=2)
    test  = generate_split(TEST_NAMES,  200,  seed=3)

    write_jsonl(f"{OUT_DIR}/train.jsonl", train)
    write_jsonl(f"{OUT_DIR}/val.jsonl",   val)
    write_jsonl(f"{OUT_DIR}/test.jsonl",  test)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = dict(sorted(collections.Counter(e["template"] for e in split).items()))
        print(f"  {name} templates: {dist}")

    print("Done. Entity sets are disjoint across splits (no name leakage).")
    print(f"  Train entities: {TRAIN_NAMES}")
    print(f"  Val   entities: {VAL_NAMES}")
    print(f"  Test  entities: {TEST_NAMES}")
