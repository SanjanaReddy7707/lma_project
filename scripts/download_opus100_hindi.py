"""
Download OPUS-100 en-hi parallel corpus and extract the Hindi side.
Source: Helsinki-NLP/opus-100 (en-hi split) on HuggingFace.
Output: data/hindi/clean/opus100_hindi.txt
"""
import re
import io
import requests
import unicodedata
import pyarrow.parquet as pq
from pathlib import Path

PARQUET_URL = (
    "https://huggingface.co/api/datasets/Helsinki-NLP/opus-100/parquet/en-hi/train/0.parquet"
)
OUT_PATH = "data/hindi/clean/opus100_hindi.txt"
DEVA_RE = re.compile(r"[ऀ-ॿ]")


def clean(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("Downloading OPUS-100 en-hi parquet (~65MB)...")
    r = requests.get(PARQUET_URL, stream=True, timeout=120)
    r.raise_for_status()
    data = bytearray()
    downloaded = 0
    for chunk in r.iter_content(65536):
        data.extend(chunk)
        downloaded += len(chunk)
        if downloaded % (10 * 1024 * 1024) == 0:
            print(f"  {downloaded // 1024 // 1024}MB")
    print(f"Downloaded {downloaded // 1024 // 1024}MB")

    table = pq.read_table(io.BytesIO(bytes(data)))
    print(f"Rows: {len(table):,}")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in table.to_pydict().get("translation", []):
            hi = row.get("hi", "") if isinstance(row, dict) else ""
            if not hi:
                skipped += 1
                continue
            hi = clean(hi)
            if len(hi) < 10:
                skipped += 1
                continue
            deva = DEVA_RE.findall(hi)
            if len(deva) / max(len(hi), 1) < 0.30:
                skipped += 1
                continue
            f.write(hi + "\n")
            written += 1

    size_mb = Path(OUT_PATH).stat().st_size / 1e6
    print(f"Written: {written:,} lines (skipped {skipped:,}) -> {OUT_PATH} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
