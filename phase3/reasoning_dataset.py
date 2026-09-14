"""
Dataset and DataLoader for reasoning fine-tuning (SFT format).

Each example is formatted as:
    प्रश्न: {input}
    उत्तर: {output}

Tokenized with SentencePiece; loss is computed only on the answer tokens
(the question tokens are masked out with -100 in the label tensor).

The DataLoader uses a custom collate_fn that pads variable-length sequences
within each batch to its maximum length.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm


class ReasoningDataset(Dataset):
    """
    Loads a JSONL file of {input, output} pairs.

    For each example the label tensor has:
      - -100 for every position that corresponds to a question token
      - actual token ids for every position that corresponds to an answer token

    This means the finetuning loss is supervised only on the answer portion
    (standard SFT masking).
    """

    PROMPT_PREFIX = "प्रश्न: "
    ANSWER_PREFIX = "\nउत्तर: "

    def __init__(self, jsonl_path, tokenizer_path, max_len=256):
        self.max_len = max_len

        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_path)

        bos = sp.bos_id()
        eos = sp.eos_id()

        self.examples = []
        skipped = 0

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)

                # Build question and answer token sequences
                q_text = f"{self.PROMPT_PREFIX}{d['input']}{self.ANSWER_PREFIX}"
                a_text = f"{d['output']}\n"

                q_ids = [bos] + sp.EncodeAsIds(q_text)   # [BOS, q0, q1, ...]
                a_ids = sp.EncodeAsIds(a_text) + [eos]    # [a0, ..., EOS]

                full_ids = q_ids + a_ids
                if len(full_ids) > max_len + 1:
                    skipped += 1
                    continue  # skip examples that exceed the model's context window

                self.examples.append({
                    "full_ids":   full_ids,
                    "prompt_len": len(q_ids),
                    "input":      d["input"],
                    "output":     d["output"],
                    "template":   d.get("template", ""),
                })

        if skipped:
            print(f"[dataset] {jsonl_path}: skipped {skipped} over-length examples")
        print(f"[dataset] {jsonl_path}: {len(self.examples)} examples loaded")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        full_ids  = e["full_ids"]     # length n = p + m
        p         = e["prompt_len"]   # length of [BOS + question tokens]

        # x  = full_ids[:-1]               — model input  (length n-1)
        # labels:
        #   positions 0 .. p-2  → -100     (question predicting itself — ignored)
        #   positions p-1 .. n-2 → full_ids[p:]  (predict each answer token)
        x      = full_ids[:-1]
        labels = [-100] * (p - 1) + full_ids[p:]

        return x, labels


def collate_reasoning(batch):
    """
    Pad a batch of variable-length (x, labels) pairs to the max sequence
    length in the batch.  Padding uses 0 for x (neutral token) and -100
    for labels (ignored by cross_entropy).
    """
    max_len = max(len(x) for x, _ in batch)
    xs, lbls = [], []
    for x, labels in batch:
        pad = max_len - len(x)
        xs.append(x + [0] * pad)
        lbls.append(labels + [-100] * pad)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(lbls, dtype=torch.long)


def make_reasoning_loader(jsonl_path, tokenizer_path, batch_size,
                           max_len=256, shuffle=True):
    ds = ReasoningDataset(jsonl_path, tokenizer_path, max_len=max_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_reasoning,
        drop_last=False,
    ), ds
