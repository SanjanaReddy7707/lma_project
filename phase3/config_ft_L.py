"""
Finetuning configuration for Model L (Nepali).

Architecture fields are identical to phase2/config_L.py.
"""

from dataclasses import dataclass


@dataclass
class FTConfig:
    # ── Identity ──────────────────────────────────────────────────────────────
    model_name: str = "Model_L"
    language:   str = "nepali"
    tokenizer_path: str = "nepali/tokenizer/nepali_bpe.model"

    # ── Architecture (must match the pretrained checkpoint exactly) ───────────
    vocab_size:  int   = 8000
    d_model:     int   = 512
    n_heads:     int   = 8
    n_layers:    int   = 6
    ffn_mult:    int   = 4
    context_len: int   = 256
    dropout:     float = 0.05
    tie_weights: bool  = True
    pos_encoding:    str = "learned"
    norm_placement:  str = "pre"

    # ── Pretrained checkpoint ─────────────────────────────────────────────────
    pretrained_ckpt: str = "phase2/checkpoints_L/best.pt"

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir:       str = "phase3/data/nepali_reasoning"
    checkpoint_dir: str = "phase3/checkpoints_ft_L"
    log_file:       str = "phase3/logs/finetune_L.jsonl"

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


CONFIG_FT_L = FTConfig()
