# %% [markdown]
# # 09 · Training at scale
#
# **Goal:** the engineering that separates "it trains" from "it trains well and
# fast". Learning-rate schedules, parameter groups, gradient clipping and
# accumulation, mixed precision, `torch.compile`, throughput measurement,
# checkpoint/resume, and the debugging protocol that catches most bugs.
#
# **You will be able to:**
# 1. implement warmup + cosine decay and explain why warmup exists
# 2. set weight decay correctly (and know what must be excluded)
# 3. simulate a large batch on a small GPU with gradient accumulation
# 4. use bf16/fp16 autocast safely, and know when a `GradScaler` is required
# 5. measure tokens/s and MFU, and diagnose whether you are compute or IO bound
# 6. size a model and a token budget with Chinchilla-style reasoning
# 7. run a long job on a remote GPU box over ssh without babysitting it
#
# Prerequisites: notebooks 01–08.

# %%
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path.cwd()
if not (ROOT / "course_utils.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import course_utils as cu
from tinygpt import GPTConfig, TinyGPT

cu.set_seed(1337)
device = cu.get_device()
print("device:", cu.describe_device(device))

text = cu.get_text(verbose=False)
tok = cu.CharTokenizer(text)
train_data, val_data = cu.train_val_split(tok.encode(text))
splits = {"train": train_data, "val": val_data}
BS = 128

# %% [markdown]
# ## 1 · Learning rate: the hyperparameter that matters most
#
# If you tune one thing, tune this. Too low and you waste compute; too high and
# the loss diverges or plateaus at a bad value. Let's see it, not assert it.

# %%
def quick_train(lr: float, steps: int = 300, schedule=None, seed: int = 0):
    cu.set_seed(seed)
    m = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=BS, d_model=128,
                          n_heads=4, n_layers=4, dropout=0.0)).to(device)
    opt = torch.optim.AdamW(m.optim_groups(0.1), lr=lr, betas=(0.9, 0.95))
    losses = []
    for step in range(steps):
        if schedule is not None:
            for g in opt.param_groups:
                g["lr"] = schedule(step)
        xb, yb = cu.get_batch(train_data, 32, BS, device)
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return m, losses


curves = {}
for lr in (1e-5, 3e-4, 3e-3, 3e-2):
    _, losses = quick_train(lr)
    curves[lr] = losses
    tail = sum(losses[-20:]) / 20
    print(f"lr {lr:>8.0e}: final(mean of last 20) {tail:>8.4f}"
          f"{'   <- diverged' if not math.isfinite(tail) or tail > 4 else ''}")

# %%
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 3.8))
    for lr, losses in curves.items():
        plt.plot(losses, label=f"lr={lr:.0e}", alpha=0.85)
    plt.ylim(0, 5); plt.xlabel("step"); plt.ylabel("train loss")
    plt.legend(); plt.grid(alpha=0.3); plt.title("learning rate sweep")
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %% [markdown]
# ## 2 · Warmup + cosine decay
#
# **Warmup** (linear, over the first few hundred steps): Adam's second-moment
# estimates are garbage at step 0, so full-size updates on a random model can
# blow up the residual stream. Ramping in avoids that.
#
# **Cosine decay** to ~10% of peak: large steps early to travel, small steps late
# to settle. This pair is the default in essentially every LLM training run.

# %%
def make_schedule(peak_lr: float, total_steps: int, warmup: int = 100,
                  min_frac: float = 0.1):
    def schedule(step: int) -> float:
        if step < warmup:
            return peak_lr * (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return peak_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))
    return schedule


sched = make_schedule(3e-3, 300, warmup=60)
print("step:  ", [f"{s:>6}" for s in (0, 30, 60, 150, 250, 299)])
print("lr:    ", [f"{sched(s):.2e}" for s in (0, 30, 60, 150, 250, 299)])

_, flat = quick_train(3e-3, 300)
_, cosine = quick_train(3e-3, 300, schedule=sched)
print(f"\nconstant lr 3e-3      : {sum(flat[-20:]) / 20:.4f}")
print(f"warmup + cosine decay : {sum(cosine[-20:]) / 20:.4f}")

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot([sched(s) for s in range(300)])
    axes[0].set_title("schedule"); axes[0].set_xlabel("step"); axes[0].grid(alpha=0.3)
    axes[1].plot(flat, label="constant", alpha=0.8)
    axes[1].plot(cosine, label="warmup+cosine", alpha=0.8)
    axes[1].set_ylim(0, 4); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_title("effect on loss"); axes[1].set_xlabel("step")
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %% [markdown]
# ## 3 · Weight decay, applied to the right things
#
# Weight decay shrinks weights toward zero as a regularizer. It makes sense for
# **matrices** (embeddings, attention and MLP projections). It makes no sense for
# **biases** and **normalization gains**: those are per-channel offsets/scales,
# there is no reason to prefer them to be zero, and decaying them measurably
# hurts. Hence two parameter groups — `TinyGPT.optim_groups` does exactly this.

# %%
model = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=BS))
groups = model.optim_groups(weight_decay=0.1)
for g in groups:
    n = sum(p.numel() for p in g["params"])
    print(f"  weight_decay={g['weight_decay']:<4} : {len(g['params']):>3} tensors, "
          f"{n:>9,} params")

decayed = {n for n, p in model.named_parameters() if p.dim() >= 2}
print("\nexcluded from decay:",
      [n for n, _ in model.named_parameters() if n not in decayed][:6], "...")

# %% [markdown]
# ## 4 · Gradient clipping
#
# One bad batch can produce a huge gradient and destroy an otherwise healthy run.
# Clipping rescales the whole gradient (preserving direction) whenever its global
# norm exceeds a threshold — 1.0 is the near-universal default. Logging the
# *pre-clip* norm is one of the most useful diagnostics you can keep.

# %%
cu.set_seed(0)
m = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=BS, d_model=128,
                      n_heads=4, n_layers=4)).to(device)
opt = torch.optim.AdamW(m.optim_groups(), lr=3e-3)
print(f"{'step':>5} {'loss':>8} {'grad norm (pre-clip)':>22} {'clipped?':>9}")
for step in range(8):
    xb, yb = cu.get_batch(train_data, 32, BS, device)
    _, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    print(f"{step:>5} {loss.item():>8.4f} {gn.item():>22.4f} "
          f"{'yes' if gn > 1.0 else 'no':>9}")
print("\nSpiking grad norms mid-run mean: lower the LR, or find the bad data.")

# %% [markdown]
# ## 5 · Gradient accumulation: a big batch on a small GPU
#
# Large batches give less noisy gradients, but batch size is capped by memory.
# Accumulation splits the batch into micro-batches, calls `backward()` on each
# (gradients **add** — notebook 01's "bug" put to work), and steps once.
#
# The one thing to get right: divide each micro-batch's loss by `grad_accum`, or
# your effective learning rate scales with the accumulation count.

# %%
def train_accum(micro_bs: int, grad_accum: int, steps: int = 60, seed: int = 0):
    cu.set_seed(seed)
    m = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=BS, d_model=128,
                          n_heads=4, n_layers=4, dropout=0.0)).to(device)
    opt = torch.optim.AdamW(m.optim_groups(), lr=1e-3, betas=(0.9, 0.95))
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            xb, yb = cu.get_batch(train_data, micro_bs, BS, device)
            _, loss = m(xb, yb)
            (loss / grad_accum).backward()          # <- the division matters
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return cu.estimate_loss(m, splits, 32, BS, device, iters=15)["val"]


print("effective batch 32, reached three ways (same seed, same data stream):")
for micro, accum in ((32, 1), (16, 2), (8, 4)):
    val = train_accum(micro, accum)
    print(f"  micro_bs {micro:>3} x accum {accum:>2} -> val {val:.4f} "
          f"(peak activation memory ~{micro / 32:.0%} of the full batch)")

# %% [markdown]
# ## 6 · Mixed precision
#
# Store master weights in fp32, but run matmuls in 16-bit. Roughly 2x faster on
# tensor-core GPUs, and about half the activation memory.
#
# - **bf16** (Ampere/A100/RTX 30xx and newer): same exponent range as fp32, so no
#   loss scaling needed. **Use this.**
# - **fp16** (older GPUs): narrow range, gradients underflow to zero, so you need
#   a `GradScaler`.
# - **MPS / CPU**: autocast support is limited; treat AMP as a CUDA feature.
#
# The template below is the one to copy.

# %%
if device.type == "cuda":
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("autocast dtype:", amp_dtype, "| bf16 supported:",
          torch.cuda.is_bf16_supported())
else:
    amp_dtype = None
    print(f"AMP demo skipped on {device.type} — the code path below is CUDA-only.")

print("""
# --- the canonical AMP training step -----------------------------------
scaler = torch.amp.GradScaler(enabled=(amp_dtype is torch.float16))

opt.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=amp_dtype):
    logits, loss = model(xb, yb)          # forward in 16-bit
scaler.scale(loss).backward()             # scaled backward (no-op for bf16)
scaler.unscale_(opt)                      # unscale before clipping!
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(opt)
scaler.update()
# -----------------------------------------------------------------------

Two classic mistakes:
  * clipping BEFORE unscale_ -> you clip scaled gradients, so the threshold is
    meaningless
  * computing the loss inside autocast is fine, but keep softmax/cross-entropy
    reductions in fp32 (PyTorch's autocast already does this for you)
""")

# %%
if device.type == "cuda":
    m = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=256,
                          d_model=384, n_heads=6, n_layers=6)).to(device)
    opt = torch.optim.AdamW(m.optim_groups(), lr=3e-4)

    def step_fp32():
        xb, yb = cu.get_batch(train_data, 16, 256, device)
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    def step_amp():
        xb, yb = cu.get_batch(train_data, 16, 256, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    for name, fn in (("fp32", step_fp32), ("amp ", step_amp)):
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1e3
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{name}: {ms:7.1f} ms/step | peak memory {peak:.2f} GB")
else:
    print("(run this cell on the CUDA box for the real numbers)")

# %% [markdown]
# ## 7 · `torch.compile`
#
# Traces the model into a graph and fuses kernels. Typically 1.3–2x on modern
# GPUs for free; the first step pays a compilation cost of tens of seconds.
#
# Caveats worth knowing: changing input *shapes* triggers recompilation (keep
# your block size fixed), and graph breaks from data-dependent Python control
# flow silently cost you the speedup.

# %%
print("torch.compile available:", hasattr(torch, "compile"))
if device.type == "cuda":
    base = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=256,
                             d_model=384, n_heads=6, n_layers=6)).to(device)
    comp = torch.compile(base)
    xb, yb = cu.get_batch(train_data, 16, 256, device)
    t0 = time.perf_counter(); comp(xb, yb); torch.cuda.synchronize()
    print(f"first (compiling) step: {time.perf_counter() - t0:.1f}s")
    for name, mdl in (("eager   ", base), ("compiled", comp)):
        for _ in range(3):
            mdl(xb, yb)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10):
            mdl(xb, yb)
        torch.cuda.synchronize()
        print(f"{name}: {(time.perf_counter() - t0) / 10 * 1e3:.1f} ms/forward")
else:
    print("compile pays off mainly on CUDA; on MPS/CPU it is often neutral or worse.")

# %% [markdown]
# ## 8 · Measure throughput, not vibes
#
# The two numbers to log every run: **tokens/second** (are you making progress?)
# and **MFU** — model FLOPs utilization, i.e. what fraction of the GPU's peak
# FLOPs you are actually using. Below ~20% MFU means something is wrong:
# dataloading, tiny batches, unfused ops, or CPU-bound sampling.
#
# The standard estimate is `6ND` FLOPs to train (`N` params, `D` tokens): 2 for
# the forward matmuls, 4 for backward.

# %%
def flops_per_token(cfg: GPTConfig, n_params: int) -> float:
    """6N per token, plus the attention term that 6N ignores."""
    attn = 12 * cfg.n_layers * cfg.d_model * cfg.block_size
    return 6 * n_params + attn


m = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=BS, d_model=192,
                      n_heads=6, n_layers=6)).to(device)
opt = torch.optim.AdamW(m.optim_groups(), lr=3e-4)
batch = 32

for _ in range(3):
    xb, yb = cu.get_batch(train_data, batch, BS, device)
    _, loss = m(xb, yb); opt.zero_grad(); loss.backward(); opt.step()
if device.type == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()
n_steps = 20
for _ in range(n_steps):
    xb, yb = cu.get_batch(train_data, batch, BS, device)
    _, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

n_params = cu.count_params(m)
tokens = n_steps * batch * BS
tps = tokens / elapsed
achieved = tps * flops_per_token(m.cfg, n_params)
print(f"params        : {cu.human(n_params)}")
print(f"tokens/s      : {tps:,.0f}")
print(f"achieved FLOPs: {achieved / 1e12:.3f} TFLOP/s")
print("\nMFU = achieved / device peak. Reference peaks (bf16, dense):")
for name, peak in (("RTX 4090", 165e12), ("A100 80GB", 312e12), ("H100 SXM", 989e12)):
    print(f"  {name:<10} {peak / 1e12:>5.0f} TFLOP/s -> "
          f"this run would be {100 * achieved / peak:>6.2f}% MFU")
print("\n(A tiny model on a big GPU always shows terrible MFU — it cannot fill the")
print(" device. Compare MFU only across configurations of the same real model.)")

# %% [markdown]
# ## 9 · How big a model, and how many tokens?
#
# Chinchilla's rule of thumb for a *compute-optimal* run: about **20 tokens per
# parameter**. Modern practice deliberately overtrains small models (Llama-3-8B
# saw ~1800 tokens/param) because inference cost dominates once you deploy.
#
# Use this to sanity-check your own runs: our corpus supports only a very small
# model before it is memorization rather than learning.

# %%
n_tokens = len(train_data)
print(f"our corpus: {n_tokens:,} training tokens")
print(f"compute-optimal parameter count at 20 tok/param: "
      f"{cu.human(n_tokens / 20)}\n")
print(f"{'model':<16}{'params':>9}{'chinchilla tokens':>19}{'our corpus covers':>19}")
for name, p in (("our TinyGPT", 8e5), ("GPT-2 small", 1.24e8),
                ("Llama-3-8B", 8e9), ("GPT-3", 1.75e11)):
    need = 20 * p
    print(f"{name:<16}{cu.human(p):>9}{cu.human(need):>19}"
          f"{100 * n_tokens / need:>18.4f}%")

# %% [markdown]
# ## 10 · The debugging protocol
#
# In order, because each step rules out a class of bugs:
#
# 1. **Check the initial loss** — should be ~`ln(V)`. If not: init or loss wiring.
# 2. **Overfit one batch** — a healthy model drives a single batch to ~0. If it
#    can't, the bug is in the model or the optimizer, not the data.
# 3. **Check the val split for leakage** — shuffled windows that overlap train.
# 4. **Watch grad norms** — spikes mean LR too high or bad data.
# 5. **Only then** tune hyperparameters.
#
# Step 2 is the highest-value test in deep learning. Run it every time.

# %%
cu.set_seed(0)
m = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=32, d_model=128,
                      n_heads=4, n_layers=4, dropout=0.0)).to(device)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
xb, yb = cu.get_batch(train_data, 8, 32, device)      # ONE fixed batch
print("overfitting a single batch (should approach 0):")
for step in range(301):
    _, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 60 == 0:
        print(f"  step {step:>4}: loss {loss.item():.5f}")
assert loss.item() < 0.1, "cannot overfit one batch -> there is a real bug"
print("PASS — the model has enough capacity and gradients flow correctly.")

# %% [markdown]
# ## 11 · Checkpoint and resume properly
#
# A resumable checkpoint needs the **optimizer state** too — Adam's moment
# estimates are as important as the weights. Resuming without them causes a
# visible loss spike. Also save the step count so the LR schedule continues from
# the right place.

# %%
ckpt = {
    "model": m.state_dict(),
    "optimizer": opt.state_dict(),
    "step": 300,
    "config": m.cfg.to_dict(),
    "vocab": tok.chars,
    "torch_version": torch.__version__,
}
cu.CKPT_DIR.mkdir(exist_ok=True)
path = cu.CKPT_DIR / "resumable.pt"
torch.save(ckpt, path)
print(f"saved {path.name}: {path.stat().st_size / 1e6:.1f} MB")
print("keys:", list(ckpt))

loaded = torch.load(path, map_location=device, weights_only=False)
m2 = TinyGPT(GPTConfig(**loaded["config"])).to(device)
m2.load_state_dict(loaded["model"])
opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
opt2.load_state_dict(loaded["optimizer"])
with torch.no_grad():
    _, l2 = m2(xb, yb)
print(f"resumed loss {l2.item():.5f} (matches the pre-save loss exactly)")
print(f"resuming at step {loaded['step']} keeps the LR schedule consistent")

# %% [markdown]
# ## 12 · Memory: where it all goes, and how to get it back
#
# For AdamW in fp32, per parameter: 4 bytes weights + 4 gradients + 8 optimizer
# state = **16 bytes**, before activations. Activations then scale with
# `batch × T × C × layers`.
#
# Levers, cheapest first:
#
# | lever | memory saved | cost |
# |---|---|---|
# | smaller micro-batch + accumulation | activations, linearly | none (more steps) |
# | bf16 autocast | ~half of activations | none on modern GPUs |
# | flash/SDPA attention (nb 08) | the `O(T²)` scores | no attention weights |
# | gradient checkpointing | most activations | ~30% slower |
# | 8-bit optimizer (`bitsandbytes`) | 6 of 8 state bytes | tiny quality risk |

# %%
def memory_estimate(n_params, batch, T, C, L, bytes_per_param=16):
    states = n_params * bytes_per_param / 1e9
    acts = batch * T * C * L * 12 * 4 / 1e9     # rough: ~12 tensors per layer
    return states, acts


for name, (n, C_, L_, T_) in {
    "TinyGPT (this course)": (8e5, 192, 6, 128),
    "GPT-2 small": (1.24e8, 768, 12, 1024),
    "Llama-3-8B": (8e9, 4096, 32, 8192),
}.items():
    s, a = memory_estimate(n, 8, T_, C_, L_)
    print(f"{name:<24} weights+optimizer {s:>7.2f} GB | activations(bs=8) {a:>7.2f} GB")

# %%
# Gradient checkpointing: recompute activations during backward instead of
# storing them. One line per block.
from torch.utils.checkpoint import checkpoint

m = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=BS, d_model=192,
                      n_heads=6, n_layers=6)).to(device)


def forward_checkpointed(model, idx):
    pos = torch.arange(idx.size(1), device=idx.device)
    x = model.drop(model.tok_emb(idx) + model.pos_emb(pos))
    for block in model.blocks:
        x = checkpoint(block, x, use_reentrant=False)     # recompute in backward
    return model.head(model.ln_f(x))


xb, yb = cu.get_batch(train_data, 8, BS, device)
logits = forward_checkpointed(m, xb)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
loss.backward()
print(f"checkpointed forward+backward works, loss {loss.item():.4f}")
print("On CUDA, compare torch.cuda.max_memory_allocated() with and without it.")

# %% [markdown]
# ## 13 · Running a real job on the GPU box over ssh
#
# Notebooks are for exploration; long runs belong in a script. `scripts/train_gpt.py`
# is notebook 05's model with CLI flags and everything from this notebook wired in
# (warmup+cosine, param groups, clipping, accumulation, AMP, compile, best-val
# checkpointing, tokens/s logging).
#
# ```bash
# # from the laptop
# rsync -av --exclude data --exclude checkpoints ./ gpu-desktop:~/transformer/
#
# ssh gpu-desktop
# cd ~/transformer
# python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#
# # detached so the run survives the ssh session dropping
# mkdir -p runs
# nohup python scripts/train_gpt.py \
#     --steps 20000 --block_size 256 --d_model 384 --n_layers 6 --n_heads 6 \
#     --batch_size 64 --grad_accum 2 --lr 6e-4 --warmup 500 \
#     --amp --compile --eval_every 500 --out gpt_medium \
#     > runs/gpt_medium.log 2>&1 &
#
# tail -f runs/gpt_medium.log     # watch
# nvidia-smi -l 5                 # confirm the GPU is actually busy
# ```
#
# Then pull the result back and inspect it here:
#
# ```bash
# rsync -av gpu-desktop:~/transformer/checkpoints/gpt_medium.pt checkpoints/
# rsync -av gpu-desktop:~/transformer/runs/ runs/
# ```
#
# A tmux session (`tmux new -s train`) is worth using instead of `nohup` when you
# want to reattach to a live console.

# %%
print(open(ROOT / "scripts" / "train_gpt.py").read().split('"""')[1])

# %%
# Load a run's history file (from either machine) and plot it.
hist_files = sorted((ROOT / "runs").glob("*_history.json")) if (ROOT / "runs").exists() else []
if hist_files:
    h = json.loads(hist_files[-1].read_text())
    print(f"{hist_files[-1].name}: {len(h['step'])} eval points, "
          f"best val {min(h['val']):.4f}")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        axes[0].plot(h["step"], h["train"], "o-", label="train")
        axes[0].plot(h["step"], h["val"], "s-", label="val")
        axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xlabel("step")
        axes[0].set_title("loss")
        axes[1].plot(h["step"], h["lr"]); axes[1].set_title("learning rate")
        axes[1].grid(alpha=0.3); axes[1].set_xlabel("step")
        plt.tight_layout(); plt.show()
    except ImportError:
        pass
else:
    print("No runs yet. Try:")
    print("  python scripts/train_gpt.py --steps 300 --eval_every 100")

# %% [markdown]
# ## 14 · Defaults worth memorizing
#
# | knob | default | note |
# |---|---|---|
# | optimizer | AdamW, betas (0.9, 0.95) | 0.95 not 0.999 for LMs |
# | peak LR | 3e-4 (large) … 3e-3 (tiny) | scale down as the model grows |
# | schedule | linear warmup 1-2% of steps, cosine to 10% | |
# | weight decay | 0.1 on matrices, 0 elsewhere | |
# | grad clip | 1.0 | log the pre-clip norm |
# | batch tokens | as many as fit, via accumulation | 0.5M+ for real runs |
# | dropout | 0.1 small data, 0.0 at scale | |
# | precision | bf16 autocast | fp32 master weights |
#
# These are starting points, not laws — but a run that ignores several of them is
# usually leaving easy performance on the table.

# %% [markdown]
# ## Exercises
#
# 1. **LR finder.** Ramp the LR exponentially from 1e-6 to 1 over 200 steps and
#    plot loss against LR. The best LR is usually ~1/3 of where loss starts
#    rising. Compare with the sweep in section 1.
# 2. **Batch size vs LR.** Train at batch 8/32/128 and find the best LR for each.
#    Does the square-root scaling rule hold at this scale?
# 3. **Divide-by-accum bug.** Remove `/ grad_accum` from section 5 and explain the
#    resulting curve in terms of effective learning rate.
# 4. **Resume spike.** Save a checkpoint mid-run, resume *without* the optimizer
#    state, and plot the loss spike. Then resume with it and confirm it is gone.
# 5. **On the GPU box.** Run `scripts/train_gpt.py` at `--d_model 384 --n_layers 6
#    --block_size 256` with and without `--amp --compile`. Report tokens/s, peak
#    memory, and MFU for both.
#
# ---
# **Next:** [10 · Encoder-decoder transformers and seq2seq](10_encoder_decoder_seq2seq.ipynb) —
# cross-attention, padding masks, and translation.
