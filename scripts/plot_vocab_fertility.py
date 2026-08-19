"""
Train tokenizers at multiple vocab sizes and plot fertility (tokens/word) vs vocab size.
Outputs: report/vocab_fertility.png
"""
import os
import sentencepiece as spm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

VOCAB_SIZES = [4000, 8000, 16000, 24000, 32000]
N_EVAL = 10000  # lines from val split to measure fertility


def train_and_eval(lang, vocab_size):
    """Train a tokenizer at given vocab size and return fertility (tokens/word) on val split."""
    train_file = f"data/{lang}/splits/train.txt"
    val_file   = f"data/{lang}/splits/val.txt"
    prefix     = f"/tmp/{lang}_vocab{vocab_size}"

    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        num_threads=4,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(f"{prefix}.model")

    total_tokens = total_words = 0
    with open(val_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_EVAL:
                break
            text = line.strip()
            if not text:
                continue
            words = text.split()
            tokens = sp.encode(text, out_type=str)
            total_words  += len(words)
            total_tokens += len(tokens)

    fertility = total_tokens / max(total_words, 1)
    print(f"  {lang} vocab={vocab_size:>5}: {fertility:.3f} tokens/word")
    return fertility


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    Path("report").mkdir(exist_ok=True)

    results = {"hindi": [], "nepali": []}

    for lang in ("hindi", "nepali"):
        print(f"\n=== {lang.upper()} ===")
        for vs in VOCAB_SIZES:
            f = train_and_eval(lang, vs)
            results[lang].append(f)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    labels = [f"{v//1000}K" for v in VOCAB_SIZES]

    for ax, lang, color in zip(axes, ("hindi", "nepali"), ("#1a5276", "#1e8449")):
        vals = results[lang]
        ax.plot(labels, vals, marker="o", color=color, linewidth=2, markersize=7)
        # mark our chosen 8K point
        idx = VOCAB_SIZES.index(8000)
        ax.scatter([labels[idx]], [vals[idx]], color="red", zorder=5, s=100,
                   label=f"Chosen: 8K ({vals[idx]:.2f})")
        ax.set_title(f"{lang.capitalize()}: Vocabulary Size vs Fertility", fontsize=13)
        ax.set_xlabel("Vocabulary Size", fontsize=11)
        ax.set_ylabel("Fertility (tokens / word)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)
        for x, y in zip(labels, vals):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color=color)

    plt.suptitle("Fertility vs Vocabulary Size — Hindi & Nepali BPE Tokenizers\n"
                 "(lower = more whole-word tokens; diminishing returns visible after 8K)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = "report/vocab_fertility.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
