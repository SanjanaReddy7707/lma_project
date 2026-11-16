#decided hyperparameters (d_model, layers, heads) to hit ~25M params
"""
Model L — Nepali decoder-only Transformer configuration.

Design choice: wider-but-shallower (6 layers, d_model=512).
Rationale: Nepali training corpus is smaller (~2.5 M lines vs 4.6 M for Hindi).
Fewer layers reduces the risk of over-fitting and shortens training time.
A wider hidden dimension (512) compensates by giving each layer more capacity
to model lexical relationships in a single pass — important because Nepali
and Hindi share many Devanagari characters but have distinct grammatical
structures that benefit from wider per-layer representations.

Parameter budget (approximate, d_model=512, n_heads=8, ffn_mult=4, n_layers=6):
  Embeddings:   8000 × 512                    =  4.10 M
  6 × block:
    Attention 4 × 512×512                     =  1.05 M/block
    FFN 512→2048→512                          =  2.10 M/block
    LayerNorms + biases                        = ~0.004 M/block
  Output head (tied to embedding)             =  0 M extra
  ─────────────────────────────────────────────
  Total ≈ 4.10 + 6 × 3.15 ≈ 23.0 M  ✓
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    # ── identity ──────────────────────────────────────────────────────────
    model_name: str = "Model_L"
    language:   str = "nepali"
    tokenizer_path: str = "nepali/tokenizer/nepali_bpe.model"

    # ── architecture ──────────────────────────────────────────────────────
    vocab_size:  int = 8000
    d_model:     int = 512
    n_heads:     int = 8        # dk = 512 / 8 = 64
    n_layers:    int = 6
    ffn_mult:    int = 4        # inner dim = 2048
    context_len: int = 256
    dropout:     float = 0.1
    tie_weights: bool = True

    # ── positional encoding choice ────────────────────────────────────────
    # "learned" → nn.Embedding of shape (context_len, d_model).
    # Constraint: same as Model H — no generalisation beyond context_len.
    pos_encoding: str = "learned"

    # ── normalization placement ───────────────────────────────────────────
    # "pre" → Pre-LN for same stability reasons as Model H.
    norm_placement: str = "pre"

    # ── training ──────────────────────────────────────────────────────────
    batch_size:    int = 64
    grad_accum:    int = 4
    max_steps:     int = 12_000
    warmup_steps:  int = 1_000
    lr:            float = 3e-4
    weight_decay:  float = 0.1
    grad_clip:     float = 1.0
    eval_interval: int = 500
    save_interval: int = 2_000
    checkpoint_dir: str = "phase2/checkpoints_L"
    log_file:       str = "phase2/logs/train_L.jsonl"


CONFIG_L = ModelConfig()
