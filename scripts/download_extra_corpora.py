"""
Download additional corpora to fill token gaps.
Hindi: IIT Bombay Hindi corpus
Nepali: IRIIS Nepali corpus, Boredoom17 Nepali corpus
"""
from datasets import load_dataset
from pathlib import Path


def try_download(dataset_name, config, out_path, text_field="text", max_rows=None):
    print(f"\nTrying {dataset_name} ({config})...")
    try:
        ds = load_dataset(dataset_name, config, split="train",
                          trust_remote_code=True, streaming=(max_rows is not None))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds:
                # Try common text field names
                text = None
                for field in [text_field, "text", "content", "sentence",
                               "translation", "hi", "ne"]:
                    if field in row and row[field]:
                        text = str(row[field]).strip()
                        break
                if text:
                    f.write(text + "\n")
                    written += 1
                if written % 100_000 == 0 and written > 0:
                    size = Path(out_path).stat().st_size / 1e6
                    print(f"  {written:,} rows | {size:.1f}MB")
                if max_rows and written >= max_rows:
                    break
        size = Path(out_path).stat().st_size / 1e6
        tokens = int(size * 1e6) // 3
        print(f"Done. {written:,} rows | {size:.1f}MB | ~{tokens:,} tokens -> {out_path}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


if __name__ == "__main__":
    # ── Nepali corpora ────────────────────────────────────────────────────
    print("=== Nepali Corpora ===")

    # IRIIS Nepali corpus (6.4M news articles, ~600M tokens)
    try_download(
        "IRIIS-RESEARCH/Nepali-Text-Corpus", None,
        "data/nepali/raw/iriis_corpus.txt"
    )

    # Boredoom17 Nepali corpus
    try_download(
        "Boredoom17/Nepali-Corpus", None,
        "data/nepali/raw/boredoom_corpus.txt"
    )

    # ── Hindi corpora ─────────────────────────────────────────────────────
    print("\n=== Hindi Corpora ===")

    # IIT Bombay Hindi monolingual
    try_download(
        "cfilt/iitb-english-hindi", None,
        "data/hindi/raw/iitb_hindi.txt",
        text_field="hi"
    )

    # AI4Bharat IndicCorp Hindi
    try_download(
        "ai4bharat/IndicCorp", "hi",
        "data/hindi/raw/indicorp_hindi.txt",
        max_rows=2_000_000
    )

    print("\nAll downloads attempted!")
