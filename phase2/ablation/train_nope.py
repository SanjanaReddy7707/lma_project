"""
OPTIONAL BONUS ABLATION — Training script for Model H without positional embeddings.

Mirrors phase2/train.py exactly; the only difference is that DecoderLM_NoPE is
imported from model_nope.py and the ablation config is used.

Usage (from lm_project/):
    python phase2/ablation/train_nope.py

Supports checkpoint resume: re-running the same command continues from the
latest checkpoint in phase2/ablation/checkpoints/.
"""

import os, sys, json, math, time

# Allow imports from lm_project/ root and phase2/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "phase2"))
sys.path.insert(0, os.path.dirname(__file__))

import torch

from model_nope      import DecoderLM_NoPE
from config_ablation import ABLATION_CONFIG as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Helpers  (copied verbatim from phase2/train.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step, cfg):
    """Linear warm-up then cosine decay to 10 % of peak LR."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.lr * 0.1
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * 0.1 + coeff * (cfg.lr - cfg.lr * 0.1)


@torch.no_grad()
def estimate_loss(model, val_loader, device, eval_batches=50):
    """Estimate val loss over eval_batches mini-batches."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_batches:
            break
        _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


def log_json(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def save_checkpoint(model, optimizer, step, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import dataclasses
    torch.save({
        "step":      step,
        "val_loss":  val_loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {"step": step},
        "config":    dataclasses.asdict(cfg),
    }, path)


def load_latest_checkpoint(ckpt_dir, model, optimizer):
    """Resume from the latest step checkpoint; return start step."""
    if not os.path.isdir(ckpt_dir):
        return 0
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir)
         if f.startswith("step_") and f.endswith(".pt")],
        key=lambda n: int(n.split("_")[1].split(".")[0]),
    )
    if not ckpts:
        return 0
    path = os.path.join(ckpt_dir, ckpts[-1])
    ck   = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    print(f"[resume] {path}  (step {ck['step']})")
    return ck["step"]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    from dataset import make_dataloader
    data_root = os.path.join(BASE_DIR, "data", cfg.language, "splits")
    tok_path  = os.path.join(BASE_DIR, cfg.tokenizer_path)
    ckpt_dir  = os.path.join(BASE_DIR, cfg.checkpoint_dir)
    log_file  = os.path.join(BASE_DIR, cfg.log_file)

    train_loader = make_dataloader(
        os.path.join(data_root, "train.txt"), tok_path,
        cfg.context_len, cfg.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = make_dataloader(
        os.path.join(data_root, "val.txt"), tok_path,
        cfg.context_len, cfg.batch_size, shuffle=False, num_workers=0,
    )
    print(f"[data] train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DecoderLM_NoPE(cfg).to(device)
    total, trainable = model.count_parameters()
    print(f"[model] {cfg.model_name}: {total:,} total  {trainable:,} trainable")
    print(f"        (Model H has {total + 256*384:,} — difference is the PE table: "
          f"{256*384:,} params)")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,    "weight_decay": cfg.weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    start_step = load_latest_checkpoint(ckpt_dir, model, optimizer)
    model.to(device)

    best_val_loss = float("inf")
    train_iter    = iter(train_loader)
    model.train()
    t0 = time.time()

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            _, loss = model(x.to(device), y.to(device))
            (loss / cfg.grad_accum).backward()
            step_loss += loss.item() / cfg.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d} | loss {step_loss:.4f} | lr {lr:.2e} | "
                  f"grad {grad_norm:.3f} | {elapsed:.1f}s")
            t0 = time.time()

        if step % cfg.eval_interval == 0 and step > 0:
            val_loss = estimate_loss(model, val_loader, device)
            val_ppl  = math.exp(val_loss)
            print(f"  [val] step {step:6d} | val_loss {val_loss:.4f} | ppl {val_ppl:.2f}")
            log_json(log_file, {
                "step": step, "train_loss": step_loss,
                "val_loss": val_loss, "val_ppl": val_ppl, "lr": lr,
            })
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss,
                                os.path.join(ckpt_dir, "best.pt"))
                print(f"  [ckpt] new best  (val_loss={val_loss:.4f})")

        if step % cfg.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, step_loss,
                            os.path.join(ckpt_dir, f"step_{step:07d}.pt"))

    # Final checkpoint
    val_loss = estimate_loss(model, val_loader, device)
    save_checkpoint(model, optimizer, cfg.max_steps, val_loss,
                    os.path.join(ckpt_dir, f"step_{cfg.max_steps:07d}_final.pt"))
    log_json(log_file, {
        "step": cfg.max_steps, "train_loss": None,
        "val_loss": val_loss, "val_ppl": math.exp(val_loss), "lr": get_lr(cfg.max_steps, cfg),
    })
    print(f"\nTraining complete — val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.2f}")


if __name__ == "__main__":
    train()
