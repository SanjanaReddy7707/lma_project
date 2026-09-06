"""
Phase 2 — Training curve plots.

Reads phase2/logs/train_H.jsonl and train_L.jsonl and produces:
  1. Training loss curve (H vs L on same axes)
  2. Validation loss curve (H vs L)
  3. Validation perplexity curve (H vs L)
  4. Learning rate schedule (H vs L)

All figures saved to report/ as PNG.

Usage:
    python phase2/plot_training.py
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_log(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def smooth(values, window=5):
    """Simple moving average for noisy training loss."""
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def make_plots(base_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    log_H = os.path.join(base_dir, "phase2", "logs", "train_H.jsonl")
    log_L = os.path.join(base_dir, "phase2", "logs", "train_L.jsonl")

    data = {}
    for key, path in [("H", log_H), ("L", log_L)]:
        if os.path.exists(path):
            data[key] = load_log(path)
            print(f"Loaded {key}: {len(data[key])} validation records")
        else:
            print(f"[warn] {path} not found — skipping Model {key}")

    if not data:
        print("No log files found.")
        return

    # ─────────────────────────────────────────────────────────────────────
    # Common style
    colors   = {"H": "#2196F3", "L": "#F44336"}
    labels   = {"H": "Model H (Hindi)", "L": "Model L (Nepali)"}
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 120})

    # ── 1. Validation loss ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, records in data.items():
        steps  = [r["step"]     for r in records]
        losses = [r["val_loss"] for r in records]
        ax.plot(steps, losses, color=colors[key], label=labels[key], linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation cross-entropy loss")
    ax.set_title("Validation Loss Curve")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "val_loss_curve.png")
    plt.savefig(path);  plt.close()
    print(f"saved: {path}")

    # ── 2. Validation perplexity ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, records in data.items():
        steps = [r["step"]    for r in records]
        ppls  = [r["val_ppl"] for r in records]
        ax.plot(steps, ppls, color=colors[key], label=labels[key], linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity (PPL)")
    ax.set_title("Validation Perplexity Curve")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "val_ppl_curve.png")
    plt.savefig(path);  plt.close()
    print(f"saved: {path}")

    # ── 3. Training loss (smoothed) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, records in data.items():
        steps  = [r["step"]       for r in records]
        losses = smooth([r["train_loss"] for r in records], window=3)
        ax.plot(steps, losses, color=colors[key], label=labels[key], linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss (smoothed)")
    ax.set_title("Training Loss Curve")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "train_loss_curve.png")
    plt.savefig(path);  plt.close()
    print(f"saved: {path}")

    # ── 4. Learning rate schedule ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, records in data.items():
        steps = [r["step"] for r in records]
        lrs   = [r["lr"]   for r in records]
        ax.plot(steps, lrs, color=colors[key], label=labels[key], linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning Rate Schedule (cosine decay with linear warmup)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "lr_schedule.png")
    plt.savefig(path);  plt.close()
    print(f"saved: {path}")

    # ── 5. Combined 2×2 summary figure ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plots = [
        (axes[0, 0], "val_loss",   "Validation Loss",        "Cross-entropy loss"),
        (axes[0, 1], "val_ppl",    "Validation Perplexity",  "PPL"),
        (axes[1, 0], "train_loss", "Training Loss (smoothed)","Loss"),
        (axes[1, 1], "lr",         "Learning Rate Schedule", "LR"),
    ]
    for ax, field, title, ylabel in plots:
        for key, records in data.items():
            steps  = [r["step"]  for r in records]
            values = [r[field]   for r in records]
            if field == "train_loss":
                values = smooth(values, window=3)
            ax.plot(steps, values, color=colors[key],
                    label=labels[key], linewidth=1.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if field == "lr":
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))

    fig.suptitle("Phase 2 Training Summary — Model H vs Model L",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "training_summary.png")
    plt.savefig(path, dpi=130);  plt.close()
    print(f"saved: {path}")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    out_dir  = os.path.join(base_dir, "report")
    make_plots(base_dir, out_dir)
