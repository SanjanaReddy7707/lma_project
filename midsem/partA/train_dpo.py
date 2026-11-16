"""
A4 + A5: DPO written from scratch, then drift (pretraining val PPL).

  L_DPO = -log σ( β [ (log πθ(y_w|x) - log πref(y_w|x)) - (log πθ(y_l|x) - log πref(y_l|x)) ] )

  * policy  = best LoRA model (merged into W0 is NOT done; we keep W0 frozen and
              continue training the LoRA A, B of the best rank)
  * reference = a frozen copy of that same model (adapters included)
  * log π(y|x) = sum of token log-probs over the reply tokens only (prompt masked)
  * β ∈ {0.1, 0.05}  (0.05 = variant 3)

Run from lm_project/:  python midsem/partA/train_dpo.py
"""
import os, sys, json, copy, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F

from common import (MID, DATA_DIR, PHASE3_FT_CKPT, PROMPT_PREFIX, ANSWER_PREFIX, get_device, get_tokenizer,
                    load_model, read_jsonl, save_json, seed_all, em_accuracy,
                    eval_three_test_sets, pretrain_val_ppl, lr_at)
from lora import apply_lora_qv

BETAS   = [0.1, 0.05]
STEPS   = 300
BATCH   = 16
LR      = 1e-4
WARMUP  = 20
LOG_EVERY = 10
RES_DIR = os.path.join(MID, "results")
CKPT_DIR = os.path.join(MID, "checkpoints")


def encode_pair(sp, prompt, reply):
    p = [sp.bos_id()] + sp.EncodeAsIds(f"{PROMPT_PREFIX}{prompt}{ANSWER_PREFIX}")
    a = sp.EncodeAsIds(f"{reply}\n") + [sp.eos_id()]   # identical to SFT format
    return p + a, len(p)


def collate(sp, batch):
    """Stack chosen and rejected into one tensor: rows [0:B]=chosen, [B:2B]=rejected."""
    seqs, plens = [], []
    for ex in batch:
        for reply in (ex["chosen"], ex["rejected"]):
            ids, pl = encode_pair(sp, ex["prompt"], reply)
            seqs.append(ids); plens.append(pl)
    order = list(range(0, 2 * len(batch), 2)) + list(range(1, 2 * len(batch), 2))
    seqs = [seqs[i] for i in order]; plens = [plens[i] for i in order]
    T = max(map(len, seqs)) - 1
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    y = torch.full((len(seqs), T), -100, dtype=torch.long)
    for i, (s, pl) in enumerate(zip(seqs, plens)):
        x[i, :len(s) - 1] = torch.tensor(s[:-1])
        y[i, pl - 1:len(s) - 1] = torch.tensor(s[pl:])     # reply tokens only
    return x, y


def seq_logprob(model, x, y):
    """Sum over reply tokens of log π(token | prefix).  Returns (2B,)."""
    logits, _ = model(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = y != -100
    tok = torch.gather(logp, 2, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(-1)


def dpo_loss(pol_w, pol_l, ref_w, ref_l, beta):
    chosen_ratio   = pol_w - ref_w          # log πθ(y_w|x)/πref(y_w|x)
    rejected_ratio = pol_l - ref_l
    margin = chosen_ratio - rejected_ratio  # the "implicit reward" margin / β
    loss = -F.logsigmoid(beta * margin).mean()
    return loss, margin.detach(), chosen_ratio.detach(), rejected_ratio.detach()


def build_policy(device, r, lora_sd):
    m, _ = load_model(device=device)
    m = apply_lora_qv(m, r=r, alpha=2 * r).to(device)
    m.load_state_dict(lora_sd, strict=False)
    return m


def run(beta, device, sp, pairs, val, best):
    seed_all(0)
    policy = build_policy(device, best["r"], best["lora"])
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    # Dropout off in the policy too: otherwise policy ≠ reference at step 0 and
    # the step-0 loss would be noisy instead of exactly log 2.
    policy.eval()
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    log = []
    t0 = time.time()
    for step in range(STEPS + 1):
        idx = torch.randint(len(pairs), (BATCH,), generator=g).tolist()
        x, y = collate(sp, [pairs[i] for i in idx]); x, y = x.to(device), y.to(device)
        with torch.no_grad():
            r_lp = seq_logprob(ref, x, y)
        p_lp = seq_logprob(policy, x, y)
        B = BATCH
        loss, margin, cr, rr = dpo_loss(p_lp[:B], p_lp[B:], r_lp[:B], r_lp[B:], beta)
        rec = {"step": step, "loss": loss.item(), "margin": margin.mean().item(),
               "chosen_logratio": cr.mean().item(), "rejected_logratio": rr.mean().item(),
               "pref_acc": (margin > 0).float().mean().item()}
        if step == 0:
            rec["note"] = "before any update"
            print(f"  β={beta}: step-0 loss {loss.item():.6f} (log 2 = 0.693147), margin {rec['margin']:.2e}")
        if step % LOG_EVERY == 0 or step == STEPS:
            log.append(rec)
        if step % 50 == 0:
            print(f"  β={beta} step {step:3d} loss {loss.item():.4f} margin {rec['margin']:.3f} "
                  f"chosen {rec['chosen_logratio']:.3f} rejected {rec['rejected_logratio']:.3f} "
                  f"({time.time()-t0:.0f}s)")
        if step == STEPS:
            break
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, STEPS, WARMUP, LR)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    tests = eval_three_test_sets(policy, sp, device)
    ppl, _ = pretrain_val_ppl(policy, device)
    torch.save({k: v.cpu() for k, v in policy.state_dict().items() if "lora_" in k},
               os.path.join(CKPT_DIR, f"dpo_beta{beta}.pt"))
    return {"beta": beta, "step0_loss": log[0]["loss"], "test": tests,
            "val_em": em_accuracy(policy, sp, val, device), "pretrain_val_ppl": ppl, "log": log}


def main():
    device = get_device(); sp = get_tokenizer()
    lora_res = json.load(open(os.path.join(RES_DIR, "partA_lora.json")))
    r = lora_res["best_rank_by_val"]
    best = torch.load(os.path.join(CKPT_DIR, f"lora_r{r}.pt"), weights_only=False)
    print(f"[DPO] starting from LoRA r={r} (best val EM {best['val_em']:.3f}, step {best['step']})")
    pairs = read_jsonl(os.path.join(MID, "data/preference_pairs.jsonl"))
    val = read_jsonl(os.path.join(DATA_DIR, "val.jsonl"))

    res = {"start_rank": r, "runs": {}}
    # ── A5 drift: base and best-LoRA perplexity ──
    base, _ = load_model(device=device)
    res["ppl_base"], _ = pretrain_val_ppl(base, device)
    ft, _ = load_model(PHASE3_FT_CKPT, device=device)
    res["ppl_phase3_full_ft"], _ = pretrain_val_ppl(ft, device)
    del base, ft
    lora_model = build_policy(device, r, best["lora"])
    res["ppl_best_lora"], _ = pretrain_val_ppl(lora_model, device)
    # mean reply log-prob of chosen/rejected under the starting policy (helps explain margins)
    del lora_model
    print(f"[A5] PPL base {res['ppl_base']:.2f} | full-FT {res['ppl_phase3_full_ft']:.2f} | "
          f"LoRA r={r} {res['ppl_best_lora']:.2f}")

    for beta in BETAS:
        res["runs"][str(beta)] = run(beta, device, sp, pairs, val, best)
        out = res["runs"][str(beta)]
        print(f"[A4] β={beta}: tests {out['test']} | PPL {out['pretrain_val_ppl']:.2f}")
        save_json(os.path.join(RES_DIR, "partA_dpo.json"), res)

    # ── figure: margin over training ──
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    colors = {"0.1": "#2a6fdb", "0.05": "#e07b39"}
    for b, run_ in res["runs"].items():
        st = [d["step"] for d in run_["log"]]
        ax[0].plot(st, [d["margin"] for d in run_["log"]], label=f"β={b}", color=colors[b], lw=2)
        ax[1].plot(st, [d["loss"] for d in run_["log"]], label=f"β={b}", color=colors[b], lw=2)
    ax[0].set_title("Mean chosen − rejected log-ratio margin")
    ax[0].set_xlabel("DPO step"); ax[0].set_ylabel("margin (nats)")
    ax[1].axhline(0.6931, ls="--", c="grey", lw=1); ax[1].text(2, 0.70, "log 2", color="grey")
    ax[1].set_title("DPO loss"); ax[1].set_xlabel("DPO step")
    for a in ax:
        a.legend(frameon=False); a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(os.path.join(MID, "figures/dpo_margin.png"), dpi=160)
    print("[saved] figures/dpo_margin.png")


if __name__ == "__main__":
    main()
