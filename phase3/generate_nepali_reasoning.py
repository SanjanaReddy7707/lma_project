"""
Synthetic Nepali reasoning dataset for fine-tuning Model L.

Mirrors the Hindi generator but uses:
  - Nepali entity names (distinct from Hindi split names)
  - Nepali possessive  को  (gender-neutral, unlike Hindi की/के/का)
  - Nepali copula  छ  (is/has)
  - Nepali comparatives  भन्दा बढी / भन्दा कम
  - Nepali question words  कसको, सबैभन्दा, मध्ये

Templates: T1–T8 (same logic as Hindi generator, different surface forms)

Output
------
phase3/data/nepali_reasoning/{train,val,test}.jsonl

Run from lm_project/
---------------------
python phase3/generate_nepali_reasoning.py
"""

import json
import random
import os
import collections

SEED = 99
random.seed(SEED)

OUT_DIR = "phase3/data/nepali_reasoning"

# ── Disjoint entity sets ──────────────────────────────────────────────────────
TRAIN_NAMES = [
    "सन्तोष", "मनिता", "राजेन्द्र", "सुमित्रा", "गोविन्द", "बिन्दु",
    "उमेश", "रेनु", "आकाश", "निशा", "प्रदीप", "शर्मिला",
    "युवराज", "दिव्या", "केशव", "लीला", "त्रिलोचन", "सरस्वती",
    "निलाम्बर", "कुसुम",
]
VAL_NAMES  = ["सुरेन्द्र", "पुष्पा", "घनश्याम", "इन्दिरा", "नवराज", "तुलसी"]
TEST_NAMES = ["रोशन", "मेनुका", "दीपक", "सावित्री", "सुभाष", "पारू"]

# ── Properties (Nepali) ───────────────────────────────────────────────────────
PROPS = [
    ("उमेर",     "वर्ष",           (10, 80)),
    ("उचाइ",    "सेन्टिमिटर",     (140, 200)),
    ("गति",     "किमि/घण्टा",     (20, 150)),
    ("मूल्य",   "रुपैयाँ",         (100, 9999)),
    ("दूरी",    "किलोमिटर",       (1, 500)),
    ("तापक्रम","डिग्री",           (5, 45)),
]

ITEMS_NE = ["आँप", "किताब", "सुन्तला", "पेन्सिल", "स्याउ"]

Q_MORE_2 = [
    "कसको {prop} बढी छ?",
    "दुईमध्ये कसको {prop} बढी छ?",
    "{n1} र {n2} मध्ये कसको {prop} बढी छ?",
]
Q_LESS_2 = [
    "कसको {prop} कम छ?",
    "दुईमध्ये कसको {prop} कम छ?",
    "{n1} र {n2} मध्ये कसको {prop} कम छ?",
]
Q_MAX_3 = [
    "सबैभन्दा बढी {prop} भएको को हो?",
    "तीनमध्ये कसको {prop} सबैभन्दा बढी छ?",
    "कसको {prop} सबैभन्दा बढी छ?",
]
Q_MIN_3 = [
    "सबैभन्दा कम {prop} भएको को हो?",
    "तीनमध्ये कसको {prop} सबैभन्दा कम छ?",
    "कसको {prop} सबैभन्दा कम छ?",
]


def vs(v, unit):
    return f"{v} {unit}".strip() if unit else str(v)


def pk(lst, rng):
    return rng.choice(lst)


def distinct_vals(lo, hi, k, rng):
    population = list(range(lo, hi + 1))
    rng.shuffle(population)
    return population[:k]


# ── Template generators (Nepali surface forms) ────────────────────────────────

def t1(names, rng):
    n1, n2 = rng.sample(names, 2)
    prop, unit, (lo, hi) = pk(PROPS, rng)
    v1, v2 = distinct_vals(lo, hi, 2, rng)

    ctx = f"{n1} को {prop} {vs(v1, unit)} छ। {n2} को {prop} {vs(v2, unit)} छ।"

    if rng.random() < 0.5:
        q   = pk(Q_MORE_2, rng).format(prop=prop, n1=n1, n2=n2)
        ans_name = n1 if v1 > v2 else n2
        ans = f"{ans_name} को {prop} बढी छ।"
    else:
        q   = pk(Q_LESS_2, rng).format(prop=prop, n1=n1, n2=n2)
        ans_name = n1 if v1 < v2 else n2
        ans = f"{ans_name} को {prop} कम छ।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T1", "property": prop}


def t2(names, rng):
    n1, n2 = rng.sample(names, 2)
    prop, unit, _ = pk(PROPS, rng)

    if rng.random() < 0.5:
        ctx = f"{n1} को {prop} {n2} भन्दा बढी छ।"
        q   = f"{n1} र {n2} मध्ये कसको {prop} कम छ?"
        ans = f"{n2} को {prop} कम छ।"
    else:
        ctx = f"{n1} को {prop} {n2} भन्दा कम छ।"
        q   = f"{n1} र {n2} मध्ये कसको {prop} बढी छ?"
        ans = f"{n2} को {prop} बढी छ।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T2", "property": prop}


def t3(names, rng):
    n1, n2, n3 = rng.sample(names, 3)
    prop, unit, (lo, hi) = pk(PROPS, rng)
    v1, v2, v3 = distinct_vals(lo, hi, 3, rng)

    ctx = (f"{n1}, {n2} र {n3} को {prop} क्रमशः "
           f"{vs(v1, unit)}, {vs(v2, unit)} र {vs(v3, unit)} छ।")

    vals = [v1, v2, v3]
    ns   = [n1, n2, n3]

    if rng.random() < 0.5:
        q   = pk(Q_MAX_3, rng).format(prop=prop)
        ans_name = ns[vals.index(max(vals))]
        ans = f"{ans_name} को {prop} सबैभन्दा बढी छ।"
    else:
        q   = pk(Q_MIN_3, rng).format(prop=prop)
        ans_name = ns[vals.index(min(vals))]
        ans = f"{ans_name} को {prop} सबैभन्दा कम छ।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T3", "property": prop}


def t4(names, rng):
    n1, n2, n3 = rng.sample(names, 3)
    prop, unit, _ = pk(PROPS, rng)

    ctx = (f"{n1} को {prop} {n2} भन्दा बढी छ। "
           f"{n2} को {prop} {n3} भन्दा बढी छ।")

    choice = rng.randint(0, 2)
    if choice == 0:
        q   = pk(Q_MAX_3, rng).format(prop=prop)
        ans = f"{n1} को {prop} सबैभन्दा बढी छ।"
    elif choice == 1:
        q   = pk(Q_MIN_3, rng).format(prop=prop)
        ans = f"{n3} को {prop} सबैभन्दा कम छ।"
    else:
        q   = f"{n1} र {n3} मध्ये कसको {prop} कम छ?"
        ans = f"{n3} को {prop} कम छ।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T4", "property": prop}


def t5(names, rng):
    n1, n2, n3 = rng.sample(names, 3)
    prop, unit, _ = pk(PROPS, rng)

    ctx = (f"{n1} को {prop} {n2} भन्दा कम छ। "
           f"{n2} को {prop} {n3} भन्दा कम छ।")

    if rng.random() < 0.5:
        q   = pk(Q_MAX_3, rng).format(prop=prop)
        ans = f"{n3} को {prop} सबैभन्दा बढी छ।"
    else:
        q   = pk(Q_MIN_3, rng).format(prop=prop)
        ans = f"{n1} को {prop} सबैभन्दा कम छ।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T5", "property": prop}


def t6(names, rng):
    n1, n2, n3 = rng.sample(names, 3)
    prop, unit, (lo, hi) = pk(PROPS, rng)
    v1, v2, v3 = distinct_vals(lo, hi, 3, rng)

    ctx = (f"{n1}, {n2} र {n3} को {prop} क्रमशः "
           f"{vs(v1, unit)}, {vs(v2, unit)} र {vs(v3, unit)} छ।")

    pairs   = sorted(zip([v1, v2, v3], [n1, n2, n3]))
    ordered = [p[1] for p in pairs]

    if rng.random() < 0.5:
        q   = f"{prop} अनुसार बढ्दो क्रममा के हो?"
        ans = f"{ordered[0]} < {ordered[1]} < {ordered[2]}"
    else:
        q   = f"{prop} अनुसार घट्दो क्रममा के हो?"
        ans = f"{ordered[2]} > {ordered[1]} > {ordered[0]}"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T6", "property": prop}


def t7(names, rng):
    n1, n2 = rng.sample(names, 2)
    prop, unit, (lo, hi) = pk(PROPS, rng)
    v = rng.randint(lo, hi)

    ctx = f"{n1} र {n2} दुवैको {prop} {vs(v, unit)} छ।"
    q   = f"दुईमध्ये कसको {prop} बढी छ?"
    ans = f"दुवैको {prop} बराबर छ।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T7", "property": prop}


def t8(names, rng):
    n1, n2 = rng.sample(names, 2)
    item = pk(ITEMS_NE, rng)
    v1 = rng.randint(1, 50)
    v2 = rng.randint(1, 50)
    while v2 == v1:
        v2 = rng.randint(1, 50)

    ctx = f"{n1} सँग {v1} वटा {item} छन्। {n2} सँग {v2} वटा {item} छन्।"

    if rng.random() < 0.5:
        q   = f"कसँग बढी {item} छन्?"
        ans_name = n1 if v1 > v2 else n2
        ans = f"{ans_name} सँग बढी {item} छन्।"
    else:
        q   = f"कसँग कम {item} छन्?"
        ans_name = n1 if v1 < v2 else n2
        ans = f"{ans_name} सँग कम {item} छन्।"

    return {"input": f"{ctx} {q}", "output": ans, "template": "T8", "property": item}


# ── Dataset builder ───────────────────────────────────────────────────────────

TEMPLATES = [t1, t2, t3, t4, t5, t6, t7, t8]
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
    print("Generating Nepali reasoning dataset …")

    train = generate_split(TRAIN_NAMES, 2000, seed=10)
    val   = generate_split(VAL_NAMES,   200,  seed=20)
    test  = generate_split(TEST_NAMES,  200,  seed=30)

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
