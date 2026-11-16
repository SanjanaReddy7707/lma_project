#wrote DataLoader to read tokenized text and create (input, target) batches
"""
Dataset utilities for causal language model pretraining.

Strategy: tokenise-once-and-cache.
  - On first call for a (text_file, tokenizer, context_len) combo, encode all
    lines with SentencePiece, pack them into a flat int32 token array, and save
    it as a .bin file next to the text file.
  - On subsequent calls, load the .bin file directly (a few seconds vs minutes).

Each Dataset item is a (context_len+1,) chunk of token ids.
The DataLoader caller splits it into input=chunk[:-1] and target=chunk[1:],
which is the standard causal LM next-token-prediction objective.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

##GENERATING CACHE FILE NAMES 
def _cache_path(text_file: str, tokenizer_path: str, context_len: int) -> str:
    tok_name = os.path.splitext(os.path.basename(tokenizer_path))[0]
    base     = os.path.splitext(text_file)[0]
    return f"{base}_{tok_name}_ctx{context_len}.bin"


def build_token_cache(text_file: str, tokenizer_path: str, context_len: int,
                      max_lines: int = None) -> str:
    """
    Tokenise text_file and write a flat int32 .bin cache.
    Returns the cache file path.
    """
    cache = _cache_path(text_file, tokenizer_path, context_len)
    if os.path.exists(cache):
        return cache

    print(f"[dataset] Building token cache: {cache}")
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    tokens = []
    with open(text_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            ids = sp.EncodeAsIds(line)
            if ids:
                tokens.append(bos_id)
                tokens.extend(ids)
                tokens.append(eos_id)
            if (i + 1) % 500_000 == 0:
                print(f"  ... {i+1:,} lines encoded, {len(tokens):,} tokens so far")

    arr = np.array(tokens, dtype=np.int32)
    arr.tofile(cache)
    print(f"[dataset] Saved {len(arr):,} tokens → {cache}")
    return cache

#Loads the .bin file and serves chunks of tokens for training.
class TokenDataset(Dataset):
    """Memory-maps a pre-built .bin token file for fast random access."""

    def __init__(self, cache_path: str, context_len: int):
        self.chunk_size = context_len + 1   # input + target share this window
        data = np.fromfile(cache_path, dtype=np.int32)
        n_chunks = len(data) // self.chunk_size
        # Trim to a multiple of chunk_size
        data = data[:n_chunks * self.chunk_size]
        # Store as a tensor for fast indexing
        self.data = torch.from_numpy(data.copy()).long()
        self.n_chunks = n_chunks

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start  = idx * self.chunk_size
        chunk  = self.data[start : start + self.chunk_size]  # (chunk_size,)
        x = chunk[:-1]   # (context_len,)  input
        y = chunk[1:]    # (context_len,)  target (shifted by 1)
        return x, y


def make_dataloader(text_file, tokenizer_path, context_len, batch_size,
                    shuffle=True, num_workers=0, max_lines=None):
    cache = build_token_cache(text_file, tokenizer_path, context_len, max_lines)
    ds    = TokenDataset(cache, context_len)
    pin   = torch.cuda.is_available()
    print(f"[dataset] {os.path.basename(text_file)}: {len(ds):,} chunks "
          f"of {context_len} tokens")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin, drop_last=True)

#quick test 
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from config_H import CONFIG_H as cfg

    base   = os.path.join(os.path.dirname(__file__), "..")
    loader = make_dataloader(
        text_file      = os.path.join(base, "data/hindi/splits/train.txt"),
        tokenizer_path = os.path.join(base, cfg.tokenizer_path),
        context_len    = cfg.context_len,
        batch_size     = cfg.batch_size,
        max_lines      = 10_000,
    )
    x, y = next(iter(loader))
    print(f"x shape: {x.shape}   y shape: {y.shape}")
    print(f"x[0,:8] = {x[0,:8].tolist()}")
    print(f"y[0,:8] = {y[0,:8].tolist()}  (should be x shifted left by 1)")
