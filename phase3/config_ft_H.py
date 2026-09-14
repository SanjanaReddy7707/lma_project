"""
Finetuning configuration for Model H (Hindi).

Architecture fields are identical to phase2/config_H.py.
Finetuning-specific fields follow.

Key differences from pretraining:
  - Lower learning rate (2e-5 vs 3e-4) to preserve pretrained weights
  - Shorter warmup and fewer total steps (dataset is small)
  - Slightly reduced dropout (0.05) to avoid under-fitting on the small SFT set
  - Loads from pretrained best.pt; saves finetuned checkpoints separately
"""

from dataclasses import dataclass, field


@dataclass
class FTConfig:
    # ── Identity ──────────────────────────────────────────────────────────────
    model_name: str = "Model_H"
    language:   str = "hindi"
    tokenizer_path: str = "hindi/tokenizer/hindi_bpe.model"

    # ── Architecture (must match the pretrained checkpoint exactly) ───────────
    vocab_size:  int   = 8000
    d_model:     int   = 384
    n_heads:     int   = 6
    n_layers:    int   = 12
    ffn_mult:    int   = 4
    context_len: int   = 256
    dropout:     float = 0.05      # lower than pretraining (0.10)
    tie_weights: bool  = True
    pos_encoding:    str = "learned"
    norm_placement:  str = "pre"

    # ── Pretrained checkpoint to start from ──────────────────────────────────
    pretrained_ckpt: str = "phase2/checkpoints_H/best.pt"

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir:       str = "phase3/data/hindi_reasoning"
    checkpoint_dir: str = "phase3/checkpoints_ft_H"
    log_file:       str = "phase3/logs/finetune_H.jsonl"

    # ── Optimiser / schedule ──────────────────────────────────────────────────
    batch_size:    int   = 32
    grad_accum:    int   = 1
    max_steps:     int   = 500
    warmup_steps:  int   = 50
    lr:            float = 2e-5
    weight_decay:  float = 0.01
    grad_clip:     float = 1.0
    eval_interval: int   = 50
    save_interval: int   = 250


CONFIG_FT_H = FTConfig()
