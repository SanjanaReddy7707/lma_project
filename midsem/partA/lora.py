"""
LoRA written from scratch (no peft).

    h = W0 x + (alpha / r) * B A x
    W0 : frozen pretrained weight  (d_out × d_in)
    A  : trainable (r × d_in),  Kaiming-uniform init (non-zero)
    B  : trainable (d_out × r), zero init  →  B A = 0 at step 0

Variant 3 adapts W_Q and W_V only.  The project model fuses Q, K, V into one
nn.Linear `qkv_proj` (d → 3d, rows [0:d]=Q, [d:2d]=K, [2d:3d]=V).  LoRALinear
therefore accepts a list of output-row slices; each slice gets its own (A, B)
pair, which is exactly equivalent to separate adapters on W_Q and W_V, while
W_K's rows receive no update.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, out_slices=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():          # freeze W0 (and bias if any)
            p.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        d_out, d_in = base.weight.shape
        self.out_slices = out_slices or [(0, d_out)]
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        for (s, e) in self.out_slices:
            A = nn.Parameter(torch.empty(r, d_in))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))     # same init nn.Linear uses
            B = nn.Parameter(torch.zeros(e - s, r))
            self.lora_A.append(A); self.lora_B.append(B)

    def delta_weight(self):
        """Full ΔW (d_out × d_in) — used for merging / inspection only."""
        d_out, d_in = self.base.weight.shape
        dW = torch.zeros(d_out, d_in, device=self.base.weight.device)
        for (s, e), A, B in zip(self.out_slices, self.lora_A, self.lora_B):
            dW[s:e] = self.scaling * (B @ A)
        return dW

    def forward(self, x):
        out = self.base(x)                                   # W0 x   (B, T, d_out)
        if len(self.out_slices) == 1 and self.out_slices[0] == (0, out.shape[-1]):
            return out + self.scaling * F.linear(F.linear(x, self.lora_A[0]), self.lora_B[0])
        parts, prev = [], 0
        for (s, e), A, B in zip(self.out_slices, self.lora_A, self.lora_B):
            if s > prev:
                parts.append(out[..., prev:s])               # un-adapted rows (W_K)
            parts.append(out[..., s:e] + self.scaling * F.linear(F.linear(x, A), B))
            prev = e
        if prev < out.shape[-1]:
            parts.append(out[..., prev:])
        return torch.cat(parts, dim=-1)


def apply_lora_qv(model, r, alpha):
    """Freeze everything, then wrap W_Q and W_V of every block with LoRA."""
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.cfg.d_model
    for blk in model.blocks:
        blk.attn.qkv_proj = LoRALinear(blk.attn.qkv_proj, r, alpha,
                                       out_slices=[(0, d), (2 * d, 3 * d)])   # Q rows, V rows
    return model


def lora_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


def count_params(model):
    """Unique parameters (tied embedding/head counted once, as in the project)."""
    seen, total, trainable = set(), 0, 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    return total, trainable
