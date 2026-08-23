#!/usr/bin/env python3
"""Train TinyGPT from the command line — the script form of notebook 05.

Designed for a real GPU box: everything is a flag, checkpoints land in
`checkpoints/`, and the CUDA-only features degrade gracefully elsewhere.

    # quick smoke test on a laptop
    python scripts/train_gpt.py --steps 200

    # a real run on the CUDA desktop
    python scripts/train_gpt.py --steps 5000 --d_model 384 --n_layers 6 \
        --n_heads 6 --block_size 256 --batch_size 64 --amp --compile

    # keep it alive over ssh
    nohup python scripts/train_gpt.py --steps 20000 --amp --compile \
        > runs/gpt.log 2>&1 &
    tail -f runs/gpt.log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import course_utils as cu
from tinygpt import GPTConfig, TinyGPT


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--data", default=None, help="path to a UTF-8 text file")
    p.add_argument("--val_frac", type=float, default=0.1)
    # model
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_tie", action="store_true", help="untie head/embedding")
    # optimization
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # runtime
    p.add_argument("--device", default=None, help="cuda | mps | cpu")
    p.add_argument("--amp", action="store_true", help="bf16/fp16 autocast")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=40)
    p.add_argument("--out", default="gpt_cli", help="checkpoint name")
    p.add_argument("--sample_tokens", type=int, default=300)
    return p.parse_args(argv)


def lr_at(step: int, args) -> float:
    """Linear warmup then cosine decay to `min_lr_frac * lr`."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    progress = min(1.0, progress)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cosine)


def main(argv=None) -> int:
    args = parse_args(argv)
    cu.set_seed(args.seed)
    device = torch.device(args.device) if args.device else cu.get_device()
    print(f"device: {cu.describe_device(device)}")

    # --- data ---------------------------------------------------------------
    text = cu.get_text(args.data)
    tok = cu.CharTokenizer(text)
    train_data, val_data = cu.train_val_split(tok.encode(text), 1 - args.val_frac)
    splits = {"train": train_data, "val": val_data}
    print(f"vocab {tok.vocab_size} | train {len(train_data):,} | val {len(val_data):,}")

    # --- model --------------------------------------------------------------
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                    d_model=args.d_model, n_heads=args.n_heads,
                    n_layers=args.n_layers, dropout=args.dropout,
                    tie_weights=not args.no_tie)
    model = TinyGPT(cfg).to(device)
    print(f"params: {cu.human(cu.count_params(model))}")

    raw_model = model
    if args.compile:
        print("compiling (first step will be slow) ...")
        model = torch.compile(model)

    opt = torch.optim.AdamW(raw_model.optim_groups(args.weight_decay),
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # bf16 needs no GradScaler; fp16 does. MPS/CPU: skip autocast entirely.
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) \
        else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype is torch.float16)
    if args.amp and not use_amp:
        print("--amp ignored: autocast here is only wired up for CUDA")
    elif use_amp:
        print(f"amp: {amp_dtype}")

    # --- train --------------------------------------------------------------
    history = {"step": [], "train": [], "val": [], "lr": []}
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    t0 = time.time()
    best_val = float("inf")

    for step in range(args.steps + 1):
        lr = lr_at(step, args)
        for group in opt.param_groups:
            group["lr"] = lr

        if step % args.eval_every == 0 or step == args.steps:
            ev = cu.estimate_loss(raw_model, splits, args.batch_size,
                                  args.block_size, device, iters=args.eval_iters)
            elapsed = time.time() - t0
            tps = step * tokens_per_step / max(elapsed, 1e-9)
            history["step"].append(step)
            history["train"].append(ev["train"])
            history["val"].append(ev["val"])
            history["lr"].append(lr)
            print(f"step {step:6d} | train {ev['train']:.4f} | val {ev['val']:.4f} "
                  f"| lr {lr:.2e} | {elapsed:6.1f}s | {tps:,.0f} tok/s")
            if ev["val"] < best_val:
                best_val = ev["val"]
                cu.save_checkpoint(raw_model, args.out, config=cfg.to_dict(),
                                   vocab=tok.chars, step=step, val_loss=best_val,
                                   args=vars(args))
        if step == args.steps:
            break

        model.train()
        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            xb, yb = cu.get_batch(train_data, args.batch_size, args.block_size, device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss = model(xb, yb)
            else:
                _, loss = model(xb, yb)
            scaler.scale(loss / args.grad_accum).backward()
        if args.grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

    # --- sample -------------------------------------------------------------
    print(f"\nbest val loss {best_val:.4f} "
          f"(bits/char {best_val / math.log(2):.3f}, ppl {math.exp(best_val):.2f})")
    if args.sample_tokens:
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = raw_model.generate(start, args.sample_tokens, temperature=0.8, top_k=20)
        print("\n--- sample ---")
        print(tok.decode(out[0]))

    hist_path = ROOT / "runs" / f"{args.out}_history.json"
    hist_path.parent.mkdir(exist_ok=True)
    hist_path.write_text(json.dumps(history, indent=1))
    print("\nhistory ->", hist_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
