"""
Quick model verification script.

Tests:
  1. Load pretrained Model H + L → generate free text from a prompt
  2. Load finetuned Model H + L  → answer a reasoning question

Run from lm_project/:
    python3 test_models.py
"""

import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))

import sentencepiece as spm
from phase2.model import DecoderLM


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path, cfg, device):
    model = DecoderLM(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"  loaded {ckpt_path}  (step={ck.get('step','?')}, val_loss={ck.get('val_loss',float('nan')):.4f})")
    return model


@torch.no_grad()
def generate(model, sp, prompt_text, cfg, device, max_new=60, temperature=0.8):
    ids = [sp.bos_id()] + sp.EncodeAsIds(prompt_text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        idx_cond = idx[:, -cfg.context_len:]
        logits, _ = model(idx_cond)
        logits = logits[0, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        if next_id == sp.eos_id():
            break
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    # decode only the generated part (after the prompt)
    gen_ids = idx[0, len(ids):].tolist()
    return sp.DecodeIds(gen_ids)


@torch.no_grad()
def answer(model, sp, question, cfg, device):
    prompt = f"प्रश्न: {question}\nउत्तर: "
    ids = [sp.bos_id()] + sp.EncodeAsIds(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    for _ in range(80):
        idx_cond = idx[:, -cfg.context_len:]
        logits, _ = model(idx_cond)
        next_id = logits[0, -1].argmax().item()
        if next_id == sp.eos_id():
            break
        generated.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    return sp.DecodeIds(generated)


# ── Device ────────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL H — Hindi
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("MODEL H — Hindi")
print("=" * 60)

from phase2.config_H import CONFIG_H
from phase3.config_ft_H import CONFIG_FT_H

sp_H = spm.SentencePieceProcessor()
sp_H.Load(CONFIG_H.tokenizer_path)

# 1. Pretrained — free text generation
print("\n[Pretrained] Free text generation:")
m_H_pre = load_model("phase2/checkpoints_H/best.pt", CONFIG_H, device)
hindi_prompt = "भारत एक विशाल देश है जहाँ"
print(f"  Prompt : {hindi_prompt}")
print(f"  Output : {generate(m_H_pre, sp_H, hindi_prompt, CONFIG_H, device)}")
del m_H_pre

# 2. Finetuned — interactive reasoning Q&A
print("\n[Finetuned] Reasoning Q&A — type a Hindi question, or press Enter to skip:")
m_H_ft = load_model("phase3/checkpoints_ft_H/best_ft.pt", CONFIG_FT_H, device)
while True:
    q = input("  Q: ").strip()
    if not q:
        break
    print(f"  A: {answer(m_H_ft, sp_H, q, CONFIG_FT_H, device)}\n")
del m_H_ft


# ══════════════════════════════════════════════════════════════════════════════
# MODEL L — Nepali
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("MODEL L — Nepali")
print("=" * 60)

from phase2.config_L import CONFIG_L
from phase3.config_ft_L import CONFIG_FT_L

sp_L = spm.SentencePieceProcessor()
sp_L.Load(CONFIG_L.tokenizer_path)

# 1. Pretrained — free text generation
print("\n[Pretrained] Free text generation:")
m_L_pre = load_model("phase2/checkpoints_L/best.pt", CONFIG_L, device)
nepali_prompt = "नेपाल एक सुन्दर देश हो जहाँ"
print(f"  Prompt : {nepali_prompt}")
print(f"  Output : {generate(m_L_pre, sp_L, nepali_prompt, CONFIG_L, device)}")
del m_L_pre

# 2. Finetuned — interactive reasoning Q&A
print("\n[Finetuned] Reasoning Q&A — type a Nepali question, or press Enter to skip:")
m_L_ft = load_model("phase3/checkpoints_ft_L/best_ft.pt", CONFIG_FT_L, device)
while True:
    q = input("  Q: ").strip()
    if not q:
        break
    print(f"  A: {answer(m_L_ft, sp_L, q, CONFIG_FT_L, device)}\n")
del m_L_ft

print("All checks passed.")
