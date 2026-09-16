#built the Transformer from scratch (InputEmbedding → CausalSelfAttention → FeedForward → TransformerBlock → DecoderLM)
"""
Decoder-only (GPT-style) Transformer language model — built from first principles.

Permitted primitives: nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout, nn.GELU.
Forbidden: nn.Transformer*, nn.MultiheadAttention, any HuggingFace model class.

Every tensor shape is annotated inline.  Symbols used throughout:
  B  – batch size
  T  – sequence length (≤ context_len)
  d  – model dimension (d_model)
  h  – number of attention heads
  dk – head dimension  (d / h)
  V  – vocabulary size
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Input representation
# ─────────────────────────────────────────────────────────────────────────────

class InputEmbedding(nn.Module):
    """Token + positional embeddings combined by addition, then dropout."""

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Learned absolute positional embeddings.
        # Shape: (context_len, d_model) — one vector per position index.
        # Constraint: the model cannot accept sequences longer than context_len
        # because there is no embedding for position >= context_len.
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.context_len = cfg.context_len

    def forward(self, idx):
        # idx: (B, T) — integer token ids
        B, T = idx.shape
        assert T <= self.context_len, (
            f"Sequence length {T} exceeds maximum context length {self.context_len}"
        )
        # positions: (1, T) broadcast over batch
        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        tok = self.tok_emb(idx)       # (B, T, d)
        pos = self.pos_emb(positions) # (1, T, d)  → broadcasts to (B, T, d)
        return self.drop(tok + pos)   # (B, T, d)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Multi-head causal self-attention
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head scaled dot-product attention with a causal (lower-triangular) mask.

    For input X ∈ (B, T, d):
      Q = X W_Q,  K = X W_K,  V = X W_V    each projected to (B, T, d)
      Reshape to (B, h, T, dk),  dk = d/h
      Attention(Q,K,V) = softmax(Q K^T / sqrt(dk) + M) V
      Concatenate heads → (B, T, d)
      Output projection W_O → (B, T, d)
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.dk      = cfg.d_model // cfg.n_heads   # per-head dimension
        self.d_model = cfg.d_model

        # Fused QKV projection — one matrix multiply instead of three.
        # Projects (B, T, d) → (B, T, 3d), then split into Q, K, V each (B, T, d).
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        # Output projection: concatenated heads (B, T, d) → (B, T, d)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Causal mask: upper-triangular filled with -inf, lower (including diagonal) with 0.
        # Registered as a buffer so it moves to the correct device with the model
        # but is not a trainable parameter.
        # Shape: (1, 1, context_len, context_len) — broadcast over B and h.
        mask = torch.triu(
            torch.full((cfg.context_len, cfg.context_len), float('-inf')), diagonal=1
        )
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # x: (B, T, d)
        B, T, d = x.shape

        # ── Q, K, V projections ────────────────────────────────────────────
        qkv = self.qkv_proj(x)              # (B, T, 3d)
        Q, K, V = qkv.split(self.d_model, dim=-1)   # each (B, T, d)

        # ── Reshape into h heads ────────────────────────────────────────────
        # (B, T, d) → (B, T, h, dk) → (B, h, T, dk)
        # The transpose makes the head dimension second so that the matmul
        # Q @ K^T operates on the (T, dk) slices for each (batch, head) pair.
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.dk).transpose(1, 2)  # (B, h, T, dk)

        Q = split_heads(Q)   # (B, h, T, dk)
        K = split_heads(K)   # (B, h, T, dk)
        V = split_heads(V)   # (B, h, T, dk)

        # ── Scaled dot-product attention ────────────────────────────────────
        # Raw scores: (B, h, T, T)
        # Division by sqrt(dk) keeps the dot-product variance at ~1 regardless
        # of dk; without it, large dk pushes softmax into near-zero-gradient
        # saturation regions.
        scale = math.sqrt(self.dk)
        scores = (Q @ K.transpose(-2, -1)) / scale   # (B, h, T, T)

        # Add causal mask: M[i,j] = -inf when j > i (future position).
        # After softmax, exp(-inf) = 0, so those positions receive zero weight.
        scores = scores + self.mask[:, :, :T, :T]    # (B, h, T, T)

        weights = F.softmax(scores, dim=-1)           # (B, h, T, T)
        # Store pre-dropout weights for attention analysis (detached, no grad).
        self.last_attn_weights = weights.detach()
        weights = self.attn_drop(weights)

        # Weighted sum of values: (B, h, T, T) @ (B, h, T, dk) → (B, h, T, dk)
        ctx = weights @ V                             # (B, h, T, dk)

        # ── Merge heads ─────────────────────────────────────────────────────
        # (B, h, T, dk) → (B, T, h, dk) → (B, T, d)
        # contiguous() is required before view because transpose creates a
        # non-contiguous memory layout.
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, d)  # (B, T, d)

        # Output projection
        out = self.out_proj(ctx)          # (B, T, d)
        return self.resid_drop(out)       # (B, T, d)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Position-wise feed-forward network
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation.
    Projects d → ffn_dim → d.  ffn_dim = ffn_mult * d (typically 4 ×).
    Applied independently to each position — the "position-wise" qualifier.
    """

    def __init__(self, cfg):
        super().__init__()
        ffn_dim = cfg.ffn_mult * cfg.d_model
        self.fc1  = nn.Linear(cfg.d_model, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # x: (B, T, d)
        x = self.fc1(x)         # (B, T, ffn_dim)
        x = F.gelu(x)           # element-wise non-linearity
        x = self.drop(x)
        x = self.fc2(x)         # (B, T, d)
        return self.drop(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Transformer block  (Pre-LayerNorm)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single decoder block using Pre-LN (LayerNorm before each sublayer).

    Pre-LN formula:
        x  ←  x + Attention(LN(x))
        x  ←  x + FFN(LN(x))

    Why Pre-LN over Post-LN?
    In Post-LN the residual stream is normalised after being added, which can
    suppress the gradient signal at depth.  Pre-LN normalises the input before
    the sublayer so gradients flow freely through the residual branch, making
    deep stacks (≥ 8 layers) significantly more stable during training.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = FeedForward(cfg)

    def forward(self, x):
        # x: (B, T, d)
        x = x + self.attn(self.ln1(x))   # residual around attention
        x = x + self.ffn(self.ln2(x))    # residual around FFN
        return x                          # (B, T, d)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full decoder-only language model
# ─────────────────────────────────────────────────────────────────────────────

class DecoderLM(nn.Module):
    """
    Decoder-only GPT-style language model.

    Forward pass stages:
      1. InputEmbedding  → (B, T, d)
      2. N × TransformerBlock
      3. Final LayerNorm  → (B, T, d)
      4. Linear head      → (B, T, V)  (next-token logits)

    Weight tying: the output linear projection shares its weight matrix with
    the token embedding matrix.  This reduces parameters by vocab_size × d_model
    (≈ 3–4 M for our setup) and acts as a regulariser — the model must learn
    a single representation space that is simultaneously useful for input lookup
    and output scoring.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed  = InputEmbedding(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.d_model)   # final normalisation
        # Output head: d → V
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: output head reuses the token embedding matrix.
        if cfg.tie_weights:
            self.head.weight = self.embed.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2-style parameter initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Scale residual projections by 1/sqrt(N) to keep residual stream variance stable.
        # Follows GPT-2 paper recommendation.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

    def forward(self, idx, targets=None):
        """
        Args:
            idx:     (B, T) token ids
            targets: (B, T) next-token ids (optional; if given, loss is returned)

        Returns:
            logits: (B, T, V)
            loss:   scalar cross-entropy (or None if targets not provided)
        """
        x = self.embed(idx)              # (B, T, d)
        for block in self.blocks:
            x = block(x)                 # (B, T, d)
        x = self.ln_f(x)                 # (B, T, d)
        logits = self.head(x)            # (B, T, V)

        loss = None
        if targets is not None:
            # Causal LM objective: predict token t+1 from position t.
            # targets are already shifted (token at position t+1) by the DataLoader.
            # Reshape: (B, T, V) → (B*T, V) and (B, T) → (B*T,) for cross_entropy.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, V)
                targets.view(-1),                    # (B*T,)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation — for qualitative inspection only.
        idx: (B, T) prompt token ids.
        """
        for _ in range(max_new_tokens):
            # Truncate to context window
            idx_cond = idx[:, -self.cfg.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # (B, V)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def count_parameters(self):
        """Return (total_params, trainable_params)."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check — run with:  python model.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from config_H import CONFIG_H
    from config_L import CONFIG_L

    for cfg, label in [(CONFIG_H, "Model H"), (CONFIG_L, "Model L")]:
        model = DecoderLM(cfg)
        total, trainable = model.count_parameters()
        print(f"\n{'='*50}")
        print(f"{label}  ({cfg.language})")
        print(f"  d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
              f"n_layers={cfg.n_layers}, ffn_mult={cfg.ffn_mult}, "
              f"context={cfg.context_len}, vocab={cfg.vocab_size}")
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,}")

        # Forward pass test
        B, T = 2, cfg.context_len
        idx     = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.randint(0, cfg.vocab_size, (B, T))
        logits, loss = model(idx, targets)
        print(f"  Forward pass:  logits {tuple(logits.shape)}, loss {loss.item():.4f}")

        # ── Causal masking verification ──────────────────────────────────
        # Changing token at position t+1 must NOT change logits at position t.
        model.eval()
        idx_a = torch.randint(0, cfg.vocab_size, (1, T))
        idx_b = idx_a.clone()
        idx_b[0, T//2] = (idx_b[0, T//2] + 1) % cfg.vocab_size  # mutate position T//2

        with torch.no_grad():
            logits_a, _ = model(idx_a)
            logits_b, _ = model(idx_b)

        # Logits at positions 0 .. T//2 - 1 must be identical
        check_pos = T//2 - 1
        max_diff = (logits_a[0, :check_pos] - logits_b[0, :check_pos]).abs().max().item()
        print(f"  Causal mask check: max logit diff at pos <{T//2} = {max_diff:.2e}  "
              f"({'PASS ✓' if max_diff < 1e-5 else 'FAIL ✗'})")
