"""
Preprocess newly scraped/downloaded files and append to clean/manual_extra.txt.
Run this after scraping completes, then re-run split_data.py.
"""
import re
import unicodedata
from pathlib import Path


def clean_line(text, lang="hi"):
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
    seen = set()
    written = skipped = bytes_read = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "a", encoding="utf-8") as fout:  # append mode
        for line in fin:
            bytes_read += len(line.encode("utf-8"))
            if byte_limit and bytes_read > byte_limit:
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
    size = Path(output_path).stat().st_size / 1e6
    est_tokens = Path(output_path).stat().st_size // 3
    print(f"  Written {written:,} (skipped {skipped:,}) -> {output_path} ({size:.1f}MB | ~{est_tokens/1e6:.1f}M tokens)")
    return written


if __name__ == "__main__":
    jobs = [
        # Hindi scraped news
        ("data/hindi/raw/manual_scraped.txt",   "data/hindi/clean/manual_extra.txt",    "hi"),
        # IIT Bombay Hindi parallel corpus (Hindi side)
        ("data/hindi/raw/iitb_hindi.txt",       "data/hindi/clean/iitb_hindi.txt",      "hi"),
        # Nepali scraped news
        ("data/nepali/raw/manual_scraped.txt",  "data/nepali/clean/manual_extra.txt",   "ne"),
        # Boredoom17 Nepali corpus
        ("data/nepali/raw/boredoom_nepali.txt", "data/nepali/clean/boredoom_nepali.txt","ne"),
    ]

    for inp, out, lang in jobs:
        if not Path(inp).exists():
            print(f"Skipping {inp} (not found)")
            continue
        if Path(inp).stat().st_size == 0:
            print(f"Skipping {inp} (empty)")
            continue
        print(f"\nProcessing {inp} ...")
        # Clear output file before writing (start fresh for extra files)
        if Path(out).exists():
            Path(out).unlink()
        process_file(inp, out, lang)

    print("\nExtra preprocessing complete!")
    print("\nNow run: python3 scripts/split_data.py")
