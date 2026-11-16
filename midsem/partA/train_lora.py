"""
A1 + A2: LoRA from scratch on W_Q, W_V (variant 3), rank sweep r ∈ {1, 4, 16}.

  * A1: parameter counts + step-0 logit-equality check
  * A2: SFT with LoRA on phase3 train.jsonl, select best step by val EM
        (same selection rule as phase3/finetune.py), report the three test
        sets, and compare with the Phase 3 full-finetune checkpoint.
        Also measures optimiser-state size and device memory for LoRA vs. a
        full-finetune run with the same batch.

Run from lm_project/:  python midsem/partA/train_lora.py
"""
import os, sys, time, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (MID, DATA_DIR, PHASE3_FT_CKPT, CONFIG_FT_H, get_device, get_tokenizer,
                    load_model, read_jsonl, save_json, seed_all, em_accuracy,
                    eval_three_test_sets, ReasoningDataset, collate_reasoning, TOK_PATH,
                    optimizer_state_bytes, device_mem_bytes, reset_peak, lr_at, pretrain_val_ppl)
from lora import apply_lora_qv, count_params, lora_state_dict

RANKS     = [1, 4, 16]
ALPHA_PER_R = 2          # alpha = 2r  →  scaling alpha/r = 2 for every rank
STEPS     = 1000
BATCH     = 32
LR        = 1e-3
WARMUP    = 50
EVAL_EVERY = 100
CKPT_DIR  = os.path.join(MID, "checkpoints")
RES_DIR   = os.path.join(MID, "results")


def sft_loss(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)


def step0_check(device, sp):
    """A1: logits of the LoRA-wrapped model must equal the pretrained model's."""
    base, _ = load_model(device=device); base.eval()
    lora, _ = load_model(device=device); lora = apply_lora_qv(lora, r=4, alpha=8).to(device); lora.eval()
    ex = read_jsonl(os.path.join(DATA_DIR, "val.jsonl"))[:8]
    ds = [sp.EncodeAsIds(e["input"]) for e in ex]
    T = max(map(len, ds))
    x = torch.tensor([d + [0] * (T - len(d)) for d in ds], device=device)
    with torch.no_grad():
        lb, _ = base(x); ll, _ = lora(x)
    return {"max_abs_logit_diff": float((lb - ll).abs().max()),
            "torch_equal": bool(torch.equal(lb, ll)),
            "max_abs_B": float(max(p.abs().max() for n, p in lora.named_parameters() if "lora_B" in n)),
            "max_abs_A": float(max(p.abs().max() for n, p in lora.named_parameters() if "lora_A" in n))}


def train_one(r, device, sp, loader, val):
    seed_all(r)
    model, cfg = load_model(device=device)
    model = apply_lora_qv(model, r=r, alpha=ALPHA_PER_R * r).to(device)
    total, trainable = count_params(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

    reset_peak(device)
    model.train()
    it, log, best = iter(loader), [], (-1, None, None)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, STEPS, WARMUP, LR)
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        loss = sft_loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1:
            mem_after_first = device_mem_bytes(device)
        if step % EVAL_EVERY == 0:
            acc = em_accuracy(model, sp, val, device)
            log.append({"step": step, "train_loss": loss.item(), "val_em": acc})
            print(f"  r={r:2d} step {step:4d} loss {loss.item():.4f} val_EM {acc:.3f} ({time.time()-t0:.0f}s)")
            if acc > best[0]:
                best = (acc, step, lora_state_dict(model))
    # restore best adapters, evaluate on three test sets
    model.load_state_dict(best[2], strict=False)
    torch.save({"r": r, "alpha": ALPHA_PER_R * r, "step": best[1], "val_em": best[0], "lora": best[2]},
               os.path.join(CKPT_DIR, f"lora_r{r}.pt"))
    tests = eval_three_test_sets(model, sp, device)
    return model, {
        "r": r, "alpha": ALPHA_PER_R * r, "trainable_params": trainable, "total_params": total,
        "trainable_share_pct": round(100 * trainable / total, 4),
        "best_step": best[1], "best_val_em": best[0], "test": tests,
        "optimizer_state_bytes": optimizer_state_bytes(opt),
        "device_mem_after_step1_bytes": mem_after_first, "log": log,
    }


def full_ft_memory(device, loader):
    """Same batch, all parameters trainable, 3 AdamW steps — memory only."""
    model, _ = load_model(device=device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    reset_peak(device)
    it = iter(loader)
    for _ in range(3):
        x, y = next(it); x, y = x.to(device), y.to(device)
        loss = sft_loss(model, x, y); opt.zero_grad(); loss.backward(); opt.step()
        if _ == 0:
            mem = device_mem_bytes(device)
    total, trainable = count_params(model)
    out = {"trainable_params": trainable, "optimizer_state_bytes": optimizer_state_bytes(opt),
           "device_mem_after_step1_bytes": mem}
    del model, opt
    return out


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = get_device(); sp = get_tokenizer()
    print("[device]", device)

    res = {"step0_check": step0_check(device, sp)}
    print("[A1] step-0 check:", res["step0_check"])

    train_ds = ReasoningDataset(os.path.join(DATA_DIR, "train.jsonl"), TOK_PATH, max_len=CONFIG_FT_H.context_len)
    g = torch.Generator().manual_seed(0)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_reasoning,
                        drop_last=True, generator=g)
    val = read_jsonl(os.path.join(DATA_DIR, "val.jsonl"))

    res["full_ft_memory"] = full_ft_memory(device, loader)
    print("[A2] full-FT memory:", res["full_ft_memory"])

    # Phase 3 full-finetune checkpoint on the same three test sets
    ft, _ = load_model(PHASE3_FT_CKPT, device=device)
    res["phase3_full_ft"] = {"test": eval_three_test_sets(ft, sp, device),
                             "val_em": em_accuracy(ft, sp, val, device)}
    print("[A2] Phase 3 full FT:", res["phase3_full_ft"])
    del ft

    base, _ = load_model(device=device)
    res["pretrained_base"] = {"test": eval_three_test_sets(base, sp, device)}
    del base

    res["lora"] = {}
    for r in RANKS:
        model, out = train_one(r, device, sp, loader, val)
        res["lora"][str(r)] = out
        print(f"[A2] r={r}: {out['test']}  trainable={out['trainable_params']:,} "
              f"({out['trainable_share_pct']}%)")
        del model
        save_json(os.path.join(RES_DIR, "partA_lora.json"), res)

    best_r = max(RANKS, key=lambda r: (res["lora"][str(r)]["best_val_em"], -r))
    res["best_rank_by_val"] = best_r
    save_json(os.path.join(RES_DIR, "partA_lora.json"), res)
    print("[A2] best rank by val EM:", best_r)


if __name__ == "__main__":
    main()
