"""
OPTIONAL BONUS ABLATION — Decoder-only Transformer WITHOUT positional embeddings.

Identical to phase2/model.py except InputEmbedding omits the positional embedding
table entirely.  The causal mask still provides weak implicit positional ordering
(earlier tokens are attended to by fewer queries than later ones), but the model
cannot learn explicit position-dependent representations.

Expected effect:
  • Higher perplexity — the model cannot distinguish "A before B" from "B before A"
  • More uniform attention patterns — heads have no positional anchor to specialise on
  • Lower Distinct-N / higher repetition — generation drifts without positional cues

This module is intentionally standalone (no imports from phase2/) so the ablation
cannot accidentally affect the graded Phase 2 models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Input representation  (positional embedding REMOVED)
# ─────────────────────────────────────────────────────────────────────────────

class InputEmbeddingNoPE(nn.Module):
    """Token embeddings only — no positional information injected.

    The model sees a set of token vectors with no indication of their order.
    Only the causal triangular mask implicitly encodes that earlier tokens have
    attended to fewer keys than later ones, but that bias is constant and weak.
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb     = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop        = nn.Dropout(cfg.dropout)
        self.context_len = cfg.context_len

    def forward(self, idx):
        # idx: (B, T)
        B, T = idx.shape
        assert T <= self.context_len
        tok = self.tok_emb(idx)   # (B, T, d)
        return self.drop(tok)     # no positional addition


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Multi-head causal self-attention  (unchanged from phase2/model.py)
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with causal mask."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.dk      = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv_proj   = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj   = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        mask = torch.triu(
            torch.full((cfg.context_len, cfg.context_len), float('-inf')), diagonal=1
        )
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, d = x.shape
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.split(self.d_model, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.dk).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        scale  = math.sqrt(self.dk)
        scores = (Q @ K.transpose(-2, -1)) / scale
        scores = scores + self.mask[:, :, :T, :T]
        weights = F.softmax(scores, dim=-1)
        self.last_attn_weights = weights.detach()
        weights = self.attn_drop(weights)

        ctx = weights @ V
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, d)
        return self.resid_drop(self.out_proj(ctx))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feed-forward network  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation: d → ffn_dim → d."""

    def __init__(self, cfg):
        super().__init__()
        ffn_dim   = cfg.ffn_mult * cfg.d_model
        self.fc1  = nn.Linear(cfg.d_model, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Transformer block  (Pre-LN, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-LN decoder block: x ← x + Attn(LN(x)); x ← x + FFN(LN(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full model  (uses InputEmbeddingNoPE instead of InputEmbedding)
# ─────────────────────────────────────────────────────────────────────────────

class DecoderLM_NoPE(nn.Module):
    """Decoder-only LM without positional embeddings (ablation variant).

    Identical to DecoderLM in phase2/model.py except the InputEmbedding is
    replaced with InputEmbeddingNoPE, which omits the positional embedding table.
    Parameter count is ~98 K smaller (context_len × d_model = 256 × 384).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg    = cfg
        self.embed  = InputEmbeddingNoPE(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.head.weight = self.embed.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2-style initialisation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * self.cfg.n_layers))

    def forward(self, idx, targets=None):
        """Returns (logits, loss).  loss is None if targets not provided."""
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x      = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation for qualitative inspection."""
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -self.cfg.context_len:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float('-inf')
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)
        return idx

    def count_parameters(self):
        """Return (total_params, trainable_params)."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
