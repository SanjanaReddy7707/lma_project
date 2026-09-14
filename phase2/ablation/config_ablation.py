"""
OPTIONAL BONUS ABLATION — Config for Model H without positional embeddings.

Identical hyperparameters to Model H (phase2/config_H.py) so the only variable
is the absence of positional embeddings.  Training is run for the same number
of steps to ensure a fair wall-clock comparison.
"""

from dataclasses import dataclass


@dataclass
class AblationConfig:
    # ── identity ──────────────────────────────────────────────────────────────
    model_name:     str = "Model_H_NoPE"
    language:       str = "hindi"
    tokenizer_path: str = "hindi/tokenizer/hindi_bpe.model"

    # ── architecture  (identical to Model H) ──────────────────────────────────
    vocab_size:  int   = 8000
    d_model:     int   = 384
    n_heads:     int   = 6
    n_layers:    int   = 12
    ffn_mult:    int   = 4
    context_len: int   = 256
    dropout:     float = 0.1
    tie_weights: bool  = True

    # ── training  (identical to Model H) ──────────────────────────────────────
    batch_size:    int   = 64
    grad_accum:    int   = 4
    max_steps:     int   = 6_500    # same as Model H best checkpoint
    warmup_steps:  int   = 1_000
    lr:            float = 3e-4
    weight_decay:  float = 0.1
    grad_clip:     float = 1.0
    eval_interval: int   = 500
    save_interval: int   = 2_000
    checkpoint_dir: str  = "phase2/ablation/checkpoints"
    log_file:       str  = "phase2/ablation/logs/train_nope.jsonl"


ABLATION_CONFIG = AblationConfig()
