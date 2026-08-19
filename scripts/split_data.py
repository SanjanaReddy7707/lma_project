import random
from pathlib import Path


def split_file(input_path, out_dir, train=0.95, val=0.025, test=0.025, seed=42):
    """Shuffle input file and write 95/2.5/2.5 train/val/test splits to out_dir."""
    print(f"\nSplitting {input_path} ...")
    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    n = len(lines)
    n_train = int(n * train)
    n_val = int(n * val)

    splits = {
        "train": lines[:n_train],
        "val":   lines[n_train:n_train + n_val],
        "test":  lines[n_train + n_val:],
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        out_path = f"{out_dir}/{name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(data)
        print(f"  {name}: {len(data):,} lines -> {out_path}")


def merge_and_split(clean_files, out_dir, lang, **kwargs):
    """Concatenate all clean_files into a single merged file, then split into train/val/test."""
    merged_path = f"/tmp/{lang}_merged.txt"
    print(f"\nMerging {len(clean_files)} files for {lang} ...")
    with open(merged_path, "w", encoding="utf-8") as fout:
        for fpath in clean_files:
            if not Path(fpath).exists():
                print(f"  Skipping missing: {fpath}")
                continue
            with open(fpath, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    split_file(merged_path, out_dir, **kwargs)


if __name__ == "__main__":
    # Hindi
    merge_and_split(
        clean_files=[
            "data/hindi/clean/mc4.txt",
            "data/hindi/clean/wikipedia.txt",
            "data/hindi/clean/wikibooks.txt",
            "data/hindi/clean/wikiquote.txt",
            "data/hindi/clean/wiktionary.txt",
            "data/hindi/clean/wikisource.txt",
            "data/hindi/clean/manual.txt",
            "data/hindi/clean/manual_extra.txt",  # scraped news
            "data/hindi/clean/iitb_hindi.txt",    # IIT Bombay parallel corpus Hindi side
            "data/hindi/clean/opus100_hindi.txt", # OPUS-100 en-hi parallel corpus Hindi side
        ],
        out_dir="data/hindi/splits",
        lang="hindi",
    )

    # Nepali
    merge_and_split(
        clean_files=[
            "data/nepali/clean/mc4.txt",
            "data/nepali/clean/wikipedia.txt",
            "data/nepali/clean/wikibooks.txt",
            "data/nepali/clean/wiktionary.txt",
            "data/nepali/clean/manual.txt",
            "data/nepali/clean/manual_extra.txt",  # scraped news
            "data/nepali/clean/boredoom_nepali.txt",  # Boredoom17 Nepali corpus
        ],
        out_dir="data/nepali/splits",
        lang="nepali",
    )

    print("\nSplits complete!")
