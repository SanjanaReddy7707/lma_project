"""Extra figures for the report (the DPO margin plot is made by train_dpo.py).
Run: python midsem/make_figures.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import MID

R = lambda f: json.load(open(os.path.join(MID, "results", f)))
lora, dpo, c1 = R("partA_lora.json"), R("partA_dpo.json"), R("c1_summary.json")
OUT = os.path.join(MID, "figures")
SETS = ["seen_entities", "unseen_entities", "unseen_wording"]
LAB = ["seen entities", "unseen entities", "unseen wording"]

# A: accuracy per model × test set
models = [("Full FT (P3)", lora["phase3_full_ft"]["test"])] + \
         [(f"LoRA r={r}", lora["lora"][r]["test"]) for r in ["1", "4", "16"]] + \
         [(f"DPO β={b}", dpo["runs"][b]["test"]) for b in ["0.1", "0.05"]]
fig, ax = plt.subplots(figsize=(9, 3.4))
w = 0.8 / len(models)
cols = ["#555555", "#9ecae1", "#4292c6", "#08519c", "#fdae6b", "#e6550d"]
for i, (name, t) in enumerate(models):
    xs = [j + (i - len(models) / 2 + .5) * w for j in range(3)]
    ax.bar(xs, [t[s] for s in SETS], w, label=name, color=cols[i])
ax.set_xticks(range(3)); ax.set_xticklabels(LAB); ax.set_ylabel("exact match"); ax.set_ylim(0, .7)
ax.legend(ncol=3, frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "partA_accuracy.png"), dpi=160); plt.close()

# C1: accuracy vs tokens
fig, ax = plt.subplots(figsize=(5.5, 3.2))
for cond, lab in [("direct", "direct"), ("cot", "CoT"), ("sc5", "SC@5")]:
    d = c1["conditions"][cond]
    ax.scatter(d["mean_gen_tokens"], d["acc_standard"], c="#2a6fdb", s=50)
    ax.scatter(d["mean_gen_tokens"], d["acc_modified"], c="#e07b39", s=50, marker="s")
    ax.annotate(lab, (d["mean_gen_tokens"], d["acc_standard"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.scatter([], [], c="#2a6fdb", label="standard (15)"); ax.scatter([], [], c="#e07b39", marker="s", label="modified (5)")
ax.set_xscale("log"); ax.set_xlabel("mean generated tokens per problem (log)"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "c1_acc_vs_tokens.png"), dpi=160); plt.close()
print("figures written")
