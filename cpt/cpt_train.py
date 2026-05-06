"""Continual pre-training: re-warm vs rehearsal.

Continues pre-training a 124M HF gpt2 checkpoint on a Python code shard while
periodically evaluating held-out NLL on (a) FineWeb webtext (the original
domain) and (b) Python code (the new domain).

Conditions are factored as schedule x replay:
  schedule  in {constant_low, rewarm_cosine}
  replay    in {0.0, 0.05}

Usage (single GPU):
  python cpt/cpt_train.py --schedule rewarm_cosine --replay 0.05 --seed 0

DDP via torchrun:
  torchrun --standalone --nproc_per_node=8 cpt/cpt_train.py \\
      --schedule rewarm_cosine --replay 0.05 --seed 0

Outputs JSON to cpt/results/{run_name}.json with the eval curve and metadata.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel


# -- shard io ------------------------------------------------------------------

MAGIC = 20240520


def load_shard(path: str | Path) -> torch.Tensor:
    """Read a uint16 .bin shard with a 256xint32 header into a CPU int64 tensor."""
    path = Path(path)
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == MAGIC, f"bad magic in {path}: {header[0]}"
    n = int(header[2])
    raw = np.fromfile(path, dtype=np.uint16, offset=256 * 4, count=n)
    return torch.from_numpy(raw.astype(np.int64))


# -- batch sampling ------------------------------------------------------------


def sample_batch(
    code_tokens: torch.Tensor,
    web_tokens: torch.Tensor | None,
    bs: int,
    ctx: int,
    replay_frac: float,
    rng: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample (bs, ctx+1) sequences. Each sequence is independently drawn from
    the rehearsal (web) or new-domain (code) buffer with prob replay_frac.

    The +1 is so we can split into (input, target) downstream.
    """
    out = torch.empty(bs, ctx + 1, dtype=torch.long, pin_memory=True)
    if replay_frac > 0 and web_tokens is not None:
        is_replay = torch.rand(bs, generator=rng) < replay_frac
    else:
        is_replay = torch.zeros(bs, dtype=torch.bool)

    for i in range(bs):
        src = web_tokens if is_replay[i] else code_tokens
        # uniform random offset
        max_off = src.size(0) - (ctx + 1) - 1
        off = int(torch.randint(0, max_off, (1,), generator=rng).item())
        out[i] = src[off : off + ctx + 1]
    return out.to(device, non_blocking=True)


# -- eval ----------------------------------------------------------------------


@torch.no_grad()
def eval_nll(model, tokens: torch.Tensor, ctx: int, device, batch_size: int = 8, max_batches: int | None = None) -> float:
    """Mean per-token NLL over a contiguous window of `tokens`.

    Uses non-overlapping windows of length ctx+1; loss is on tokens 1..ctx of
    each window predicted from tokens 0..ctx-1.
    """
    model.eval()
    n_windows = tokens.size(0) // (ctx + 1)
    n_windows = min(n_windows, (max_batches or 10**9) * batch_size)
    losses = []
    counts = []
    for i in range(0, n_windows, batch_size):
        j = min(i + batch_size, n_windows)
        starts = [k * (ctx + 1) for k in range(i, j)]
        bat = torch.stack([tokens[s : s + ctx + 1] for s in starts]).to(device, non_blocking=True)
        x, y = bat[:, :-1], bat[:, 1:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum")
        losses.append(loss.item())
        counts.append(y.numel())
    model.train()
    total_loss = sum(losses)
    total_count = sum(counts)
    return total_loss / max(total_count, 1)


# -- schedules -----------------------------------------------------------------


def lr_at(step: int, total_steps: int, schedule: str, peak_lr: float, min_lr: float, warmup: int) -> float:
    if schedule == "constant_low":
        return min_lr
    if schedule == "rewarm_cosine":
        if step < warmup:
            return peak_lr * (step + 1) / max(warmup, 1)
        # cosine from peak_lr down to min_lr over remaining steps
        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
    raise ValueError(schedule)


# -- main ----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    # condition
    ap.add_argument("--schedule", choices=["constant_low", "rewarm_cosine"], required=True)
    ap.add_argument("--replay", type=float, default=0.0, help="rehearsal fraction in [0, 1]")
    ap.add_argument("--seed", type=int, default=0)
    # data
    ap.add_argument("--code_train", default="cpt/data/code_train.bin")
    ap.add_argument("--code_val", default="cpt/data/code_val.bin")
    ap.add_argument("--web_val", default="data/fineweb10B/fineweb_val_000000.bin")
    ap.add_argument("--web_train", default="data/fineweb10B/fineweb_train_000001.bin")
    # model
    ap.add_argument("--model", default="gpt2")
    # training
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--per_device_bs", type=int, default=8)
    ap.add_argument("--peak_lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # eval
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=8)
    # io
    ap.add_argument("--out_dir", default="cpt/results")
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    # -- DDP setup
    is_dist = "RANK" in os.environ
    if is_dist:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        rank, world_size, local_rank = 0, 1, 0
        torch.cuda.set_device(0)
    device = torch.device("cuda", local_rank)
    is_master = rank == 0

    # -- seeds (each rank gets a distinct stream, but determinism within rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    rng = torch.Generator().manual_seed(args.seed * 100003 + rank)

    # -- run name & output
    run_name = args.run_name or f"{args.schedule}_replay{args.replay:.2f}_seed{args.seed}"
    out_path = Path(args.out_dir) / f"{run_name}.json"
    if is_master:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[rank0] run_name={run_name}  out={out_path}", flush=True)

    # -- data
    code_train = load_shard(args.code_train)
    code_val = load_shard(args.code_val)
    web_train = load_shard(args.web_train) if args.replay > 0 else None
    web_val = load_shard(args.web_val)

    if is_master:
        print(
            f"[rank0] code_train={code_train.numel():,}  code_val={code_val.numel():,}  "
            f"web_train={web_train.numel() if web_train is not None else 0:,}  "
            f"web_val={web_val.numel():,}",
            flush=True,
        )

    # -- model
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    model.gradient_checkpointing_disable()
    if is_dist:
        ddp_model = DDP(model, device_ids=[local_rank])
        unwrapped = ddp_model.module
    else:
        ddp_model = model
        unwrapped = model

    # AdamW with no decay on biases / layernorm / 1d params
    decay, no_decay = [], []
    for n, p in unwrapped.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or n.endswith(".bias") else decay).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.min_lr,
        betas=(0.9, 0.95),
    )

    # -- baseline eval (step 0, before any updates)
    def do_eval():
        ddp_model.eval()
        web_nll = eval_nll(unwrapped, web_val, args.ctx, device, args.eval_bs, args.eval_batches)
        code_nll = eval_nll(unwrapped, code_val, args.ctx, device, args.eval_bs, args.eval_batches)
        ddp_model.train()
        if is_dist:
            t = torch.tensor([web_nll, code_nll], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            web_nll, code_nll = t[0].item(), t[1].item()
        return web_nll, code_nll

    history: list[dict] = []
    web_nll0, code_nll0 = do_eval()
    if is_master:
        print(f"[rank0] step=0   web_nll={web_nll0:.4f}  code_nll={code_nll0:.4f}", flush=True)
        history.append({"step": 0, "web_nll": web_nll0, "code_nll": code_nll0, "lr": float(optim.param_groups[0]["lr"])})

    # -- training loop
    ddp_model.train()
    t_start = time.time()
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.schedule, args.peak_lr, args.min_lr, args.warmup)
        for g in optim.param_groups:
            g["lr"] = lr

        bat = sample_batch(code_train, web_train, args.per_device_bs, args.ctx, args.replay, rng, device)
        x, y = bat[:, :-1], bat[:, 1:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = ddp_model(x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unwrapped.parameters(), args.grad_clip)
        optim.step()

        if step % args.eval_every == 0 or step == args.steps:
            web_nll, code_nll = do_eval()
            elapsed = time.time() - t_start
            if is_master:
                print(
                    f"[rank0] step={step:4d}  lr={lr:.2e}  loss={loss.item():.3f}  "
                    f"web_nll={web_nll:.4f} (Δ={web_nll - web_nll0:+.3f})  "
                    f"code_nll={code_nll:.4f} (Δ={code_nll - code_nll0:+.3f})  "
                    f"t={elapsed:.0f}s",
                    flush=True,
                )
                history.append({"step": step, "web_nll": web_nll, "code_nll": code_nll, "lr": lr, "train_loss": loss.item()})

    # -- save
    if is_master:
        result = {
            "run_name": run_name,
            "config": vars(args),
            "world_size": world_size,
            "tokens_per_step": args.per_device_bs * args.ctx * world_size,
            "history": history,
            "baseline_web_nll": web_nll0,
            "baseline_code_nll": code_nll0,
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[rank0] wrote {out_path}", flush=True)

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
