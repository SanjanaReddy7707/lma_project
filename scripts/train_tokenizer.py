import sentencepiece as spm
from pathlib import Path


def train_tokenizer(input_files, model_prefix, vocab_size, lang):
    """Train a SentencePiece BPE tokenizer on the given input files and save model + vocab."""
    combined = f"/tmp/{lang}_tokenizer_input.txt"
    print(f"\nCombining input files for {lang} tokenizer...")
    total_lines = 0
    with open(combined, "w", encoding="utf-8") as fout:
        for fpath in input_files:
            if not Path(fpath).exists():
                print(f"  Skipping missing: {fpath}")
                continue
            with open(fpath, encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
                        total_lines += 1
    print(f"  Total lines for training: {total_lines:,}")

    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    print(f"Training {lang} BPE tokenizer (vocab={vocab_size})...")

    spm.SentencePieceTrainer.train(
        input=combined,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        num_threads=4,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
    )
    print(f"Saved: {model_prefix}.model and {model_prefix}.vocab")


EXAMPLES = {
    "hindi": [
        "नमस्ते दुनिया",
        "भारत एक विविधताओं से भरा देश है",
        "आज मौसम बहुत अच्छा है",
    ],
    "nepali": [
        "नमस्ते संसार",
        "नेपाल एक सुन्दर देश हो",
        "आज मौसम राम्रो छ",
    ],
}


def evaluate_tokenizer(model_path, test_file, lang="hindi", n_samples=5000):
    """Compute fertility (chars/token) and UNK rate on n_samples lines from test_file."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    total_tokens = total_chars = unk_count = 0
    with open(test_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            text = line.strip()
            if not text:
                continue
            tokens = sp.encode(text, out_type=str)
            total_tokens += len(tokens)
            total_chars += len(text)
            unk_count += tokens.count("<unk>")

    avg_chars_per_token = total_chars / max(total_tokens, 1)
    unk_rate = unk_count / max(total_tokens, 1)
    print(f"\n  Avg chars/token: {avg_chars_per_token:.2f}")
    print(f"  UNK rate:        {unk_rate:.4%}")

    examples = EXAMPLES.get(lang, EXAMPLES["hindi"])
    print("  Tokenization examples:")
    for ex in examples:
        print(f"    '{ex}' -> {sp.encode(ex, out_type=str)}")


if __name__ == "__main__":
    # Hindi tokenizer — 8K vocab
    train_tokenizer(
        input_files=[
            "data/hindi/splits/train.txt",
        ],
        model_prefix="hindi/tokenizer/hindi_bpe",
        vocab_size=8000,
        lang="hindi",
    )
    evaluate_tokenizer(
        "hindi/tokenizer/hindi_bpe.model",
        "data/hindi/splits/val.txt",
        lang="hindi",
    )

    # Nepali tokenizer — 8K vocab
    train_tokenizer(
        input_files=[
            "data/nepali/splits/train.txt",
        ],
        model_prefix="nepali/tokenizer/nepali_bpe",
        vocab_size=8000,
        lang="nepali",
    )
    evaluate_tokenizer(
        "nepali/tokenizer/nepali_bpe.model",
        "data/nepali/splits/val.txt",
        lang="nepali",
    )

    print("\nAll tokenizers trained!")
