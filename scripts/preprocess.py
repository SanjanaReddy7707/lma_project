# cleaned all raw text (removed URLs, filtered non-Devanagari, deduplicated) → saved to data/hindi/clean/ and data/nepali/clean/
import re
import unicodedata
from pathlib import Path

MC4_BYTE_LIMIT = 1_600_000_000  # 1.6 GB of mc4 = ~500M tokens


def clean_line(text, lang="hi"):
    """Normalize, strip URLs/emails, and discard lines with <30% Devanagari characters."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if lang in ("hi", "ne"):
        devanagari = sum(1 for c in text if 'ऀ' <= c <= 'ॿ')
        if len(text) > 0 and devanagari / len(text) < 0.3:
            return ""
    return text


def process_file(input_path, output_path, lang="hi", byte_limit=None):
    """Read input_path line-by-line, clean each line, deduplicate, and write to output_path."""
    seen = set()
    written = skipped = bytes_read = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            bytes_read += len(line.encode("utf-8"))
            if byte_limit and bytes_read > byte_limit:
                print(f"  Reached byte limit ({byte_limit/1e9:.1f}GB), stopping.")
                break
            cleaned = clean_line(line.strip(), lang)
            if not cleaned or len(cleaned) < 20:
                skipped += 1
                continue
            key = cleaned[:80]
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            fout.write(cleaned + "\n")
            written += 1
            if written % 500_000 == 0:
                print(f"  {written:,} lines written...")
    print(f"  Written: {written:,}  Skipped: {skipped:,}  -> {output_path}")
    return written


def count_tokens_estimate(file_path):
    """Estimate token count as total character count ÷ 3 (Devanagari averages ~3 chars/token)."""
    chars = sum(len(line) for line in open(file_path, encoding="utf-8"))
    return chars // 3


if __name__ == "__main__":
    # mc4: capped at 1.6GB to stay near 500M tokens
    # All other files: process fully (they are small)
    jobs = [
        ("data/hindi/raw/mc4.txt",          "data/hindi/clean/mc4.txt",          "hi", MC4_BYTE_LIMIT),
        ("data/hindi/raw/wikipedia.txt",     "data/hindi/clean/wikipedia.txt",    "hi", None),
        ("data/hindi/raw/wiktionary.txt",    "data/hindi/clean/wiktionary.txt",   "hi", None),
        ("data/hindi/raw/wikibooks.txt",     "data/hindi/clean/wikibooks.txt",    "hi", None),
        ("data/hindi/raw/wikiquote.txt",     "data/hindi/clean/wikiquote.txt",    "hi", None),
        ("data/hindi/raw/wikisource.txt",    "data/hindi/clean/wikisource.txt",   "hi", None),
        ("data/hindi/raw/manual.txt",        "data/hindi/clean/manual.txt",       "hi", None),
        ("data/nepali/raw/mc4.txt",          "data/nepali/clean/mc4.txt",         "ne", MC4_BYTE_LIMIT),
        ("data/nepali/raw/wikipedia.txt",    "data/nepali/clean/wikipedia.txt",   "ne", None),
        ("data/nepali/raw/wiktionary.txt",   "data/nepali/clean/wiktionary.txt",  "ne", None),
        ("data/nepali/raw/wikibooks.txt",    "data/nepali/clean/wikibooks.txt",   "ne", None),
        ("data/nepali/raw/manual.txt",       "data/nepali/clean/manual.txt",      "ne", None),
    ]

    for inp, out, lang, limit in jobs:
        if not Path(inp).exists():
            print(f"Skipping {inp} (not found)")
            continue
        if Path(inp).stat().st_size == 0:
            print(f"Skipping {inp} (empty)")
            continue
        print(f"\nProcessing {inp} ...")
        n = process_file(inp, out, lang, byte_limit=limit)
        est = count_tokens_estimate(out)
        print(f"  Estimated tokens: {est:,}")

    print("\nPreprocessing complete!")
