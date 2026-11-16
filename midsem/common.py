"""
Shared helpers for the make-up mid-sem (variant 3, Model H = Hindi).

Everything here reuses the project code unchanged:
  phase2/model.py            DecoderLM (from-scratch transformer)
  phase3/config_ft_H.py      architecture + SFT settings
  phase3/reasoning_dataset   "प्रश्न: ... \nउत्तर: ..." SFT format with answer-only loss
"""
import os, sys, json, math, random
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # lm_project/
MID  = os.path.join(BASE, "midsem")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "phase3"))
sys.path.insert(0, os.path.join(BASE, "phase2"))

import sentencepiece as spm
from phase2.model import DecoderLM
from config_ft_H import CONFIG_FT_H
from reasoning_dataset import ReasoningDataset, collate_reasoning

PRETRAINED_CKPT = os.path.join(BASE, "phase2/checkpoints_H/best.pt")
PHASE3_FT_CKPT  = os.path.join(BASE, "phase3/checkpoints_ft_H/best_ft.pt")
TOK_PATH        = os.path.join(BASE, CONFIG_FT_H.tokenizer_path)
DATA_DIR        = os.path.join(BASE, CONFIG_FT_H.data_dir)
VAL_BIN         = os.path.join(BASE, "data/hindi/splits/val_hindi_bpe_ctx256.bin")

TEST_SETS = {   # name -> jsonl path
    "seen_entities":   os.path.join(MID, "data/test_seen_entities.jsonl"),
    "unseen_entities": os.path.join(DATA_DIR, "test.jsonl"),          # original Phase 3 test set
    "unseen_wording":  os.path.join(MID, "data/test_unseen_wording.jsonl"),
}

PROMPT_PREFIX = "प्रश्न: "
ANSWER_PREFIX = "\nउत्तर: "


def seed_all(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.Load(TOK_PATH)
    return sp


def load_model(ckpt=PRETRAINED_CKPT, device="cpu", dropout=None):
    cfg = CONFIG_FT_H
    if dropout is not None:
        import dataclasses
        cfg = dataclasses.replace(cfg, dropout=dropout)
    model = DecoderLM(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return model.to(device), cfg


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── Exact-match evaluation (identical rule to phase3/eval_reasoning.py) ──────

def normalize(text):
    return text.strip().rstrip("।\n").strip()


@torch.no_grad()
def greedy_answer(model, sp, question, device, max_new=80):
    ids = [sp.bos_id()] + sp.EncodeAsIds(f"{PROMPT_PREFIX}{question}{ANSWER_PREFIX}")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.cfg.context_len:])
        nxt = int(logits[0, -1].argmax())
        if nxt == sp.eos_id():
            break
        out.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    return sp.DecodeIds(out)


@torch.no_grad()
def em_accuracy(model, sp, examples, device, keep_samples=0):
    was_training = model.training
    model.eval()
    correct, samples = 0, []
    for ex in examples:
        pred = normalize(greedy_answer(model, sp, ex["input"], device))
        ok = pred == normalize(ex["output"])
        correct += ok
        if len(samples) < keep_samples:
            samples.append({"input": ex["input"], "gold": ex["output"], "pred": pred, "em": ok})
    if was_training:
        model.train()
    acc = correct / max(len(examples), 1)
    return (acc, samples) if keep_samples else acc


def eval_three_test_sets(model, sp, device):
    return {name: round(em_accuracy(model, sp, read_jsonl(p), device), 4)
            for name, p in TEST_SETS.items()}


# ── Pretraining-validation perplexity (same protocol as phase2/eval_metrics.py:
#     first 200 non-shuffled batches of 64 × 256-token chunks of the val split) ──

@torch.no_grad()
def pretrain_val_ppl(model, device, batch_size=64, max_batches=200):
    was_training = model.training
    model.eval()
    T = model.cfg.context_len
    data = np.fromfile(VAL_BIN, dtype=np.int32)
    n_chunks = len(data) // (T + 1)
    data = torch.from_numpy(data[: n_chunks * (T + 1)].astype(np.int64)).view(n_chunks, T + 1)
    tot, n = 0.0, 0
    for i in range(min(max_batches, n_chunks // batch_size)):
        chunk = data[i * batch_size:(i + 1) * batch_size].to(device)
        # .contiguous(): model.py flattens targets with .view(), which needs contiguous memory
        _, loss = model(chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous())
        tot += loss.item(); n += 1
    if was_training:
        model.train()
    return math.exp(tot / n), tot / n


# ── Memory accounting ────────────────────────────────────────────────────────

def device_mem_bytes(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if device.type == "mps":
        torch.mps.empty_cache()


def optimizer_state_bytes(opt):
    return sum(t.numel() * t.element_size()
               for st in opt.state.values() for t in st.values() if torch.is_tensor(t))


def lr_at(step, max_steps, warmup, peak):
    """Linear warm-up then cosine decay to 10 % (same shape as phase3/finetune.py)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, max_steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))
