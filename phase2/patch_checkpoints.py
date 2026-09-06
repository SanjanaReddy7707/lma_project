"""
One-time script: add 'config' and 'scheduler' keys to existing checkpoints
that were saved before the updated save_checkpoint function.
"""
import sys, os, dataclasses, torch
sys.path.insert(0, os.path.dirname(__file__))

def patch(ckpt_path, cfg):
    ck = torch.load(ckpt_path, map_location="cpu")
    if "config" in ck:
        print(f"  already patched: {ckpt_path}")
        return
    ck["config"]    = dataclasses.asdict(cfg)
    ck["scheduler"] = {"step": ck["step"]}
    torch.save(ck, ckpt_path)
    print(f"  patched: {ckpt_path}")

from config_H import CONFIG_H
from config_L import CONFIG_L

base = os.path.join(os.path.dirname(__file__), "..")
for cfg, subdir in [(CONFIG_H, "checkpoints_H"), (CONFIG_L, "checkpoints_L")]:
    d = os.path.join(base, "phase2", subdir)
    for fname in os.listdir(d):
        if fname.endswith(".pt"):
            patch(os.path.join(d, fname), cfg)

print("Done.")
