#downloaded mC4 + Wikipedia for Hindi and Nepali → saved to data/hindi/raw/ and data/nepali/raw/
from datasets import load_dataset
from pathlib import Path


def download_mc4(lang, out_path):
    print(f"\nDownloading mC4 {lang}...")
    ds = load_dataset("mc4", lang, split="train", streaming=True, trust_remote_code=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example["text"].strip()
            if text:
                f.write(text + "\n")
            written += 1
            if written % 100_000 == 0:
                print(f"  {written:,} docs...")
    print(f"Done. {written:,} docs -> {out_path}")


def download_wikipedia(lang, out_path):
    print(f"\nDownloading Wikipedia {lang}...")
    ds = load_dataset("wikipedia", f"20231101.{lang}",
                      split="train", trust_remote_code=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if text:
                f.write(text + "\n")
    print(f"Done. {len(ds):,} articles -> {out_path}")


if __name__ == "__main__":
    # Hindi (Model H)
    download_mc4("hi", "data/hindi/raw/mc4.txt")
    download_wikipedia("hi", "data/hindi/raw/wikipedia.txt")

    # Nepali (Model L)
    download_mc4("ne", "data/nepali/raw/mc4.txt")
    download_wikipedia("ne", "data/nepali/raw/wikipedia.txt")

    print("\nAll downloads complete!")
