"""
Pretraining script for Model H and Model L.

Usage:
    python phase2/train.py --model H   # pretrain on Hindi
    python phase2/train.py --model L   # pretrain on Nepali

Features:
    - AdamW optimiser with cosine LR decay + linear warm-up
    - Gradient accumulation for large effective batch size
    - Gradient norm clipping
    - Periodic validation loss + perplexity logging
    - Checkpoint saving (best val loss + every save_interval steps)
    - MPS / CUDA / CPU device selection
    - Resumes from latest checkpoint if one exists
"""

import os, sys, json, math, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step, cfg):
    """
    Linear warm-up then cosine decay to 10% of peak LR.
    Matches the schedule used in GPT-2 / NanoGPT.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.lr * 0.1
    # cosine phase
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * 0.1 + coeff * (cfg.lr - cfg.lr * 0.1)


@torch.no_grad()
def estimate_loss(model, val_loader, device, eval_batches=50):
    """Estimate validation loss over eval_batches mini-batches."""
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    for _ in range(eval_batches):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


def log_json(log_file, record):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def save_checkpoint(model, optimizer, step, val_loss, path, cfg=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step":      step,
        "val_loss":  val_loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # Scheduler state: LR is computed analytically from step + cfg,
        # so saving the step and cfg is equivalent to saving scheduler state.
        "scheduler": {"step": step},
    }
    if cfg is not None:
        import dataclasses
        payload["config"] = dataclasses.asdict(cfg)
    torch.save(payload, path)


def load_latest_checkpoint(checkpoint_dir, model, optimizer):
    """Return the step number from the latest checkpoint, or 0 if none."""
    if not os.path.isdir(checkpoint_dir):
        return 0
    ckpts = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith("step_") and f.endswith(".pt")],
        key=lambda n: int(n.split("_")[1].split(".")[0])
    )
    if not ckpts:
        return 0
    path = os.path.join(checkpoint_dir, ckpts[-1])
    ck   = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    print(f"[resume] Loaded checkpoint {path} at step {ck['step']}")
    return ck["step"]


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg, base_dir):
    # ── Device selection ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    from dataset import make_dataloader
    data_root   = os.path.join(base_dir, "data", cfg.language, "splits")
    tok_path    = os.path.join(base_dir, cfg.tokenizer_path)
    ckpt_dir    = os.path.join(base_dir, cfg.checkpoint_dir)
    log_file    = os.path.join(base_dir, cfg.log_file)

    train_loader = make_dataloader(
        os.path.join(data_root, "train.txt"), tok_path,
        cfg.context_len, cfg.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = make_dataloader(
        os.path.join(data_root, "val.txt"), tok_path,
        cfg.context_len, cfg.batch_size, shuffle=False, num_workers=0,
    )
    print(f"[data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    from model import DecoderLM
    model = DecoderLM(cfg).to(device)
    total, trainable = model.count_parameters()
    print(f"[model] {cfg.model_name}: {total:,} total params, {trainable:,} trainable")

    # ── Optimiser ─────────────────────────────────────────────────────────
    # Separate weight-decay groups: no decay for biases and LayerNorm parameters.
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,    "weight_decay": cfg.weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = load_latest_checkpoint(ckpt_dir, model, optimizer)
    model.to(device)  # re-move after potential CPU load

    best_val_loss = float('inf')
    train_iter    = iter(train_loader)
    accum_loss    = 0.0

    model.train()
    t0 = time.time()

    for step in range(start_step, cfg.max_steps):
        # ── Learning rate update ──────────────────────────────────────────
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ─────────────────────────────────────────
        optimizer.zero_grad()
        step_loss = 0.0
        for micro in range(cfg.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            loss    = loss / cfg.grad_accum   # scale for accumulation
            loss.backward()
            step_loss += loss.item()

        # ── Gradient clipping + optimiser step ────────────────────────────
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────
        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d} | loss {step_loss:.4f} | lr {lr:.2e} | "
                  f"grad_norm {grad_norm:.3f} | {elapsed:.1f}s")
            t0 = time.time()

        # ── Validation ───────────────────────────────────────────────────
        if step % cfg.eval_interval == 0 and step > 0:
            val_loss = estimate_loss(model, val_loader, device)
            val_ppl  = math.exp(val_loss)
            print(f"  [val] step {step} | val_loss {val_loss:.4f} | ppl {val_ppl:.2f}")
            log_json(log_file, {
                "step": step, "train_loss": step_loss,
                "val_loss": val_loss, "val_ppl": val_ppl, "lr": lr,
            })
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss,
                                 os.path.join(ckpt_dir, "best.pt"), cfg=cfg)
                print(f"  [ckpt] saved best checkpoint (val_loss={val_loss:.4f})")

        # ── Periodic save ─────────────────────────────────────────────────
        if step % cfg.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, step_loss,
                             os.path.join(ckpt_dir, f"step_{step:07d}.pt"), cfg=cfg)

    # Final checkpoint
    val_loss = estimate_loss(model, val_loader, device)
    save_checkpoint(model, optimizer, cfg.max_steps, val_loss,
                     os.path.join(ckpt_dir, f"step_{cfg.max_steps:07d}_final.pt"), cfg=cfg)
    print(f"\nTraining complete. Final val_loss={val_loss:.4f}, ppl={math.exp(val_loss):.2f}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["H", "L"], required=True,
                        help="Which model to train: H (Hindi) or L (Nepali)")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..")

    if args.model == "H":
        from config_H import CONFIG_H as cfg
    else:
        from config_L import CONFIG_L as cfg

    train(cfg, base_dir)
