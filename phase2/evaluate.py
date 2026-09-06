"""
Evaluation script for a trained DecoderLM checkpoint.

Computes:
  1. Perplexity on val and test splits.
  2. Parameter count table.
  3. Causal masking verification (no future leakage).
  4. Sample generation for qualitative inspection.

Usage:
    python phase2/evaluate.py --model H --checkpoint phase2/checkpoints_H/best.pt
    python phase2/evaluate.py --model L --checkpoint phase2/checkpoints_L/best.pt

    # Evaluate before training (random weights) to confirm perplexity ≈ vocab_size
    python phase2/evaluate.py --model H --no-checkpoint
"""

import os, sys, math, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch


def perplexity(model, loader, device, max_batches=200):
    """Average cross-entropy loss → perplexity on a DataLoader."""
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            n += 1
    avg_loss = total_loss / n if n > 0 else float('inf')
    return avg_loss, math.exp(avg_loss)


def verify_causal_mask(model, cfg, device):
    """
    Empirical causal masking test.

    Mutates token at position mid in a sequence and checks that logits at
    all positions < mid are unchanged.  If the model leaks future information,
    the logits before mid would differ between the two inputs.
    """
    model.eval()
    T   = cfg.context_len
    mid = T // 2

    idx_a = torch.randint(0, cfg.vocab_size, (1, T), device=device)
    idx_b = idx_a.clone()
    # Mutate position mid (so positions 0..mid-1 should be unaffected)
    idx_b[0, mid] = (idx_b[0, mid] + 1) % cfg.vocab_size

    with torch.no_grad():
        logits_a, _ = model(idx_a)
        logits_b, _ = model(idx_b)

    max_diff = (logits_a[0, :mid] - logits_b[0, :mid]).abs().max().item()
    passed   = max_diff < 1e-5
    print(f"  Causal mask test: max |logit diff| at positions <{mid} = {max_diff:.2e}  "
          f"[{'PASS' if passed else 'FAIL'}]")
    if not passed:
        print("  WARNING: future tokens affect earlier logits — causal masking is broken!")
    return passed


def sample_generation(model, cfg, tokenizer_path, device, prompt_text="",
                       max_new=50, temperature=0.8, top_k=40):
    """Generate a short sample from the model for qualitative inspection."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    if prompt_text:
        ids = [sp.bos_id()] + sp.EncodeAsIds(prompt_text)
    else:
        ids = [sp.bos_id()]

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new, temperature=temperature, top_k=top_k)
    token_ids = out[0].tolist()
    text = sp.DecodeIds(token_ids)
    return text


# ─────────────────────────────────────────────────────────────────────────────

def evaluate(cfg, checkpoint_path, base_dir, max_lines_eval=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}\n")

    from model import DecoderLM
    from dataset import make_dataloader

    model = DecoderLM(cfg).to(device)

    if checkpoint_path:
        ck = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"[checkpoint] loaded from {checkpoint_path} (step {ck.get('step','?')})\n")
    else:
        print("[checkpoint] none — evaluating random initialisation\n")

    # ── Parameter report ─────────────────────────────────────────────────
    total, trainable = model.count_parameters()
    print("=" * 56)
    print(f"Model:            {cfg.model_name} ({cfg.language})")
    print(f"d_model:          {cfg.d_model}")
    print(f"n_layers:         {cfg.n_layers}")
    print(f"n_heads:          {cfg.n_heads}")
    print(f"ffn_mult:         {cfg.ffn_mult}  (inner dim {cfg.ffn_mult * cfg.d_model})")
    print(f"context_len:      {cfg.context_len}")
    print(f"vocab_size:       {cfg.vocab_size}")
    print(f"weight tying:     {cfg.tie_weights}")
    print(f"pos encoding:     {cfg.pos_encoding}")
    print(f"norm placement:   {cfg.norm_placement}-norm")
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print("=" * 56)

    # Break down parameters by component
    print("\nParameter breakdown:")
    tok_emb_p = sum(p.numel() for n, p in model.named_parameters()
                    if "tok_emb" in n)
    pos_emb_p = sum(p.numel() for n, p in model.named_parameters()
                    if "pos_emb" in n)
    attn_p    = sum(p.numel() for n, p in model.named_parameters()
                    if "attn" in n or "qkv" in n or "out_proj" in n)
    ffn_p     = sum(p.numel() for n, p in model.named_parameters()
                    if "ffn" in n or "fc1" in n or "fc2" in n)
    other_p   = total - tok_emb_p - pos_emb_p - attn_p - ffn_p
    for name, count in [("  Token embedding", tok_emb_p),
                         ("  Pos  embedding",  pos_emb_p),
                         ("  Attention layers", attn_p),
                         ("  FFN layers",       ffn_p),
                         ("  Other (LN, head)", other_p)]:
        print(f"  {name:<22}: {count:>10,}")

    # ── Causal mask verification ──────────────────────────────────────────
    print("\nCausal masking verification:")
    verify_causal_mask(model, cfg, device)

    # ── Perplexity on val and test ─────────────────────────────────────────
    data_root  = os.path.join(base_dir, "data", cfg.language, "splits")
    tok_path   = os.path.join(base_dir, cfg.tokenizer_path)

    print("\nPerplexity evaluation:")
    for split in ["val", "test"]:
        split_file = os.path.join(data_root, f"{split}.txt")
        if not os.path.exists(split_file):
            print(f"  {split}: file not found, skipping")
            continue
        loader = make_dataloader(split_file, tok_path, cfg.context_len,
                                  batch_size=cfg.batch_size, shuffle=False,
                                  max_lines=max_lines_eval)
        avg_loss, ppl = perplexity(model, loader, device)
        print(f"  {split:<5}: loss = {avg_loss:.4f},  perplexity = {ppl:.2f}")

    # ── Sample generation ─────────────────────────────────────────────────
    print("\nSample generation (temperature=0.8, top_k=40):")
    if checkpoint_path:
        gen_text = sample_generation(model, cfg, tok_path, device,
                                      prompt_text="", max_new=80)
        print(f"  {gen_text[:300]}")
    else:
        print("  (skipped — random weights produce nonsense)")

    print()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["H", "L"], required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt checkpoint file (omit for random init eval)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Explicitly skip checkpoint loading")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Cap lines per split for fast testing (e.g. 2000)")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..")

    if args.model == "H":
        from config_H import CONFIG_H as cfg
    else:
        from config_L import CONFIG_L as cfg

    # Resolve checkpoint path
    ckpt = None
    if args.no_checkpoint:
        ckpt = None
    elif args.checkpoint:
        ckpt = args.checkpoint
    else:
        # auto-detect best checkpoint
        auto = os.path.join(base_dir, cfg.checkpoint_dir, "best.pt")
        ckpt = auto if os.path.exists(auto) else None

    evaluate(cfg, ckpt, base_dir, max_lines_eval=args.max_lines)
