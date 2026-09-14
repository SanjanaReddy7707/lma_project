"""
Supervised fine-tuning (SFT) of a pretrained decoder-only LM on reasoning tasks.

Key design choices
------------------
* Loads from a pretrained best.pt checkpoint; does NOT share data or tokenizer
  between models (each language is fine-tuned independently).
* SFT loss: cross-entropy computed only over answer tokens (question tokens
  are masked with -100 and ignored by PyTorch's cross_entropy).
* Lower learning rate (2e-5) and short cosine decay to preserve pretrained
  representations while adapting to the Q&A format.
* Validates after every eval_interval steps using both NLL loss and
  greedy exact-match accuracy on held-out val examples.
* Saves checkpoints in the same resume-capable format as pretraining.

Usage
-----
    python phase3/finetune.py --model H
    python phase3/finetune.py --model L
"""

import os, sys, json, math, argparse, time, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import sentencepiece as spm


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_lr(step, cfg):
    """Linear warm-up → cosine decay to 10 % of peak LR."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.lr * 0.1
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * 0.1 + coeff * (cfg.lr - cfg.lr * 0.1)


def log_json(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def save_checkpoint(model, optimizer, step, val_loss, val_acc, path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step":      step,
        "val_loss":  val_loss,
        "val_acc":   val_acc,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config":    dataclasses.asdict(cfg),
    }, path)


def load_pretrained(path, model):
    """Load model weights from a pretraining checkpoint (ignore optimizer)."""
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    step = ck.get("step", 0)
    vloss = ck.get("val_loss", float("inf"))
    print(f"[load] pretrained step={step}, val_loss={vloss:.4f}")


def load_ft_checkpoint(ckpt_dir, model, optimizer):
    """Resume from the latest SFT checkpoint if one exists; return start step."""
    if not os.path.isdir(ckpt_dir):
        return 0
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt")],
        key=lambda n: int(n.split("_")[1].split(".")[0]),
    )
    if not ckpts:
        return 0
    path = os.path.join(ckpt_dir, ckpts[-1])
    ck   = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    print(f"[resume] SFT checkpoint {path} at step {ck['step']}")
    return ck["step"]


# ── Evaluation ────────────────────────────────────────────────────────────────

PROMPT_PREFIX = "प्रश्न: "
ANSWER_PREFIX = "\nउत्तर: "


@torch.no_grad()
def greedy_generate(model, prompt_ids, max_new, eos_id, device, cfg):
    """Greedy generation; stops at EOS or max_new tokens."""
    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    for _ in range(max_new):
        idx_cond = idx[:, -cfg.context_len:]
        logits, _ = model(idx_cond)
        next_id = logits[0, -1].argmax().item()
        if next_id == eos_id:
            break
        generated.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    return generated


def normalize(text):
    """Strip whitespace and trailing punctuation for comparison."""
    return text.strip().rstrip("।\n").strip()


@torch.no_grad()
def evaluate(model, val_loader, val_ds, device, cfg, sp, n_gen=100):
    """
    Returns (avg_loss, exact_match_accuracy, entity_accuracy).

    exact_match: normalized predicted answer == normalized gold answer
    entity_acc : gold entity name appears in prediction (more lenient)
    """
    model.eval()

    # ── NLL loss on val set ───────────────────────────────────────────────────
    total_loss, n_batches = 0.0, 0
    for x, labels in val_loader:
        x, labels = x.to(device), labels.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        total_loss += loss.item()
        n_batches  += 1
    avg_loss = total_loss / max(n_batches, 1)

    # ── Exact-match accuracy on a subset of val examples ─────────────────────
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    em_correct, ent_correct, n_eval = 0, 0, 0
    for ex in val_ds.examples[:n_gen]:
        q_text   = f"{PROMPT_PREFIX}{ex['input']}{ANSWER_PREFIX}"
        prompt   = [bos_id] + sp.EncodeAsIds(q_text)
        gold_ans = normalize(ex["output"])

        gen_ids = greedy_generate(model, prompt, max_new=80, eos_id=eos_id,
                                  device=device, cfg=cfg)
        pred_ans = normalize(sp.DecodeIds(gen_ids))

        if pred_ans == gold_ans:
            em_correct += 1

        # Entity accuracy: first word of gold answer should appear in prediction
        gold_first = gold_ans.split()[0] if gold_ans else ""
        if gold_first and gold_first in pred_ans:
            ent_correct += 1

        n_eval += 1

    em_acc  = em_correct  / max(n_eval, 1)
    ent_acc = ent_correct / max(n_eval, 1)

    model.train()
    return avg_loss, em_acc, ent_acc


# ── Main training loop ────────────────────────────────────────────────────────

def finetune(cfg, base_dir):
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = os.path.join(base_dir, cfg.tokenizer_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)

    # ── Data ──────────────────────────────────────────────────────────────────
    from reasoning_dataset import make_reasoning_loader

    data_dir = os.path.join(base_dir, cfg.data_dir)
    train_loader, _ = make_reasoning_loader(
        os.path.join(data_dir, "train.jsonl"), tok_path,
        cfg.batch_size, max_len=cfg.context_len, shuffle=True,
    )
    val_loader, val_ds = make_reasoning_loader(
        os.path.join(data_dir, "val.jsonl"), tok_path,
        cfg.batch_size, max_len=cfg.context_len, shuffle=False,
    )
    print(f"[data] train batches/epoch: {len(train_loader)}, "
          f"val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from phase2.model import DecoderLM
    model = DecoderLM(cfg).to(device)

    # Load pretrained weights
    pretrained_path = os.path.join(base_dir, cfg.pretrained_ckpt)
    load_pretrained(pretrained_path, model)
    model.to(device)

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

    # Resume from SFT checkpoint if one exists
    ckpt_dir   = os.path.join(base_dir, cfg.checkpoint_dir)
    log_file   = os.path.join(base_dir, cfg.log_file)
    start_step = load_ft_checkpoint(ckpt_dir, model, optimizer)
    model.to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc  = -1.0
    best_val_loss = float("inf")
    train_iter = iter(train_loader)
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
                x, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, labels  = next(train_iter)

            x, labels = x.to(device), labels.to(device)
            logits, _ = model(x)

            # SFT loss: ignore question tokens (-100 positions)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            ) / cfg.grad_accum
            loss.backward()
            step_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"step {step:5d} | loss {step_loss:.4f} | lr {lr:.2e} | "
                  f"grad {grad_norm:.3f} | {elapsed:.1f}s")
            t0 = time.time()

        # ── Validation ────────────────────────────────────────────────────────
        if step % cfg.eval_interval == 0 and step > 0:
            val_loss, em_acc, ent_acc = evaluate(
                model, val_loader, val_ds, device, cfg, sp, n_gen=100,
            )
            print(f"  [val] step {step:5d} | loss {val_loss:.4f} | "
                  f"EM {em_acc:.3f} | Ent {ent_acc:.3f}")
            log_json(log_file, {
                "step": step, "train_loss": step_loss,
                "val_loss": val_loss, "em_acc": em_acc,
                "ent_acc": ent_acc, "lr": lr,
            })
            if em_acc > best_val_acc or (em_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc  = em_acc
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss, em_acc,
                                os.path.join(ckpt_dir, "best_ft.pt"), cfg)
                print(f"  [ckpt] saved best (EM={em_acc:.3f})")

        if step % cfg.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, step_loss, -1,
                            os.path.join(ckpt_dir, f"step_{step:06d}.pt"), cfg)

    # Final evaluation
    val_loss, em_acc, ent_acc = evaluate(
        model, val_loader, val_ds, device, cfg, sp, n_gen=100,
    )
    save_checkpoint(model, optimizer, cfg.max_steps, val_loss, em_acc,
                    os.path.join(ckpt_dir, f"step_{cfg.max_steps:06d}_final.pt"), cfg)
    log_json(log_file, {
        "step": cfg.max_steps, "train_loss": None,
        "val_loss": val_loss, "em_acc": em_acc,
        "ent_acc": ent_acc, "lr": get_lr(cfg.max_steps, cfg),
    })
    print(f"\nFinetuning complete — val_loss={val_loss:.4f}, "
          f"EM={em_acc:.3f}, Ent={ent_acc:.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["H", "L"], required=True)
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..")

    if args.model == "H":
        from config_ft_H import CONFIG_FT_H as cfg
    else:
        from config_ft_L import CONFIG_FT_L as cfg

    finetune(cfg, base_dir)
