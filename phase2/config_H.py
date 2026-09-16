#decided hyperparameters (d_model, layers, heads) to hit ~25M params
"""
Model H — Hindi decoder-only Transformer configuration.

Design choice: deeper-and-narrower (12 layers, d_model=512).
Rationale: Hindi has rich morphology; deeper stacks let higher layers build
on abstract morphological representations learned in lower layers.  A narrower
hidden dimension keeps the total parameter budget near 25 M while still giving
12 opportunities to compose representations.

Pre-norm (LayerNorm before each sublayer) is used throughout because it
produces smoother gradient magnitudes across depth, making 12-layer training
more stable without learning-rate warm-up tricks.

Parameter budget (approximate):
  Embeddings:   vocab(8000) × d(512)          =  4.10 M
  12 × block:
    Attention Q/K/V/O: 4 × 512×512            =  1.05 M/block
    FFN (512→2048→512): 512×2048 + 2048×512   =  2.10 M/block
    LayerNorms + biases                        = ~0.004 M/block
  Output head (tied to embedding)             =  0 M extra
  ─────────────────────────────────────────────
  Total ≈ 4.10 + 12 × 3.15 ≈ 41.9 M  → too large; reduce heads/ffn_mult.

Revised with d_model=384, n_heads=6, ffn_mult=4, n_layers=12, context=256:
  Embeddings:   8000 × 384                    =  3.07 M
  12 × block:
    Attention 4 × 384×384                     =  0.59 M/block
    FFN 384→1536→384                          =  1.18 M/block
  Total ≈ 3.07 + 12 × 1.77 ≈ 24.3 M  ✓
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    # ── identity ──────────────────────────────────────────────────────────
    model_name: str = "Model_H"
    language:   str = "hindi"
    tokenizer_path: str = "hindi/tokenizer/hindi_bpe.model"

    # ── architecture ──────────────────────────────────────────────────────
    vocab_size:  int = 8000
    d_model:     int = 384      # hidden / embedding dimension
    n_heads:     int = 6        # must divide d_model; dk = d_model / n_heads = 64
    n_layers:    int = 12       # transformer blocks stacked
    ffn_mult:    int = 4        # inner FFN dim = ffn_mult * d_model = 1536
    context_len: int = 256      # maximum sequence length (constrains positional embeddings)
    dropout:     float = 0.1
    tie_weights: bool = True    # share input-embedding and output-projection weights

    # ── positional encoding choice ────────────────────────────────────────
    # "learned" → nn.Embedding of shape (context_len, d_model); trained end-to-end.
    # Constraint: model cannot generalise to sequences longer than context_len
    # because there is no embedding for position >= context_len.
    pos_encoding: str = "learned"

    # ── normalization placement ───────────────────────────────────────────
    # "pre" → LayerNorm applied before each sublayer (Pre-LN / GPT-2 style).
    # More stable for deep stacks: gradients flow through the residual path
    # without passing through a LayerNorm, reducing exploding/vanishing risk.
    norm_placement: str = "pre"

    # ── training ──────────────────────────────────────────────────────────
    batch_size:    int = 64
    grad_accum:    int = 4        # effective batch = batch_size * grad_accum = 256
    max_steps:     int = 20_000
    warmup_steps:  int = 1_000
    lr:            float = 3e-4
    weight_decay:  float = 0.1
    grad_clip:     float = 1.0
    eval_interval: int = 500
    save_interval: int = 2_000
    checkpoint_dir: str = "phase2/checkpoints_H"
    log_file:       str = "phase2/logs/train_H.jsonl"


CONFIG_H = ModelConfig()
