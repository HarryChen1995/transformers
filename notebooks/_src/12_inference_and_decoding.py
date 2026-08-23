# %% [markdown]
# # 12 · Inference: KV cache, decoding strategies and quantization
#
# **Goal:** training is over; now make the model *fast* and its output *good*.
# We implement a KV cache from scratch (the single biggest inference win),
# compare every decoding strategy in use, implement beam search, and quantize
# weights to int8 by hand.
#
# **You will be able to:**
# 1. explain why naive generation is `O(T²)` and fix it with a cache
# 2. distinguish prefill from decode, and know which is compute vs memory bound
# 3. choose between greedy / temperature / top-k / top-p / min-p with reasons
# 4. implement beam search and say when it helps and when it hurts
# 5. quantize weights to int8 and measure the size/quality trade
#
# Prerequisites: notebooks 01–05, 08 (GQA).

# %%
import math
import sys
import time
from dataclasses import dataclass, asdict
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

# %% [markdown]
# ## 1 · A model to serve
#
# Load notebook 05's checkpoint if it exists, otherwise train a quick one, so
# this notebook stands alone.

# %%
cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=128, d_model=192,
                n_heads=6, n_layers=6, dropout=0.1, flash=False)
model = TinyGPT(cfg).to(device)

ckpt_path = cu.CKPT_DIR / "tinygpt.pt"
loaded = False
if ckpt_path.exists():
    ck = cu.load_checkpoint("tinygpt")
    if ck["config"].get("d_model") == cfg.d_model and \
       ck["config"].get("vocab_size") == cfg.vocab_size:
        model.load_state_dict(ck["state_dict"], strict=False)
        loaded = True
        print("loaded checkpoint from notebook 05")
if not loaded:
    print("no matching checkpoint; training briefly")
    cu.train(model, splits, steps=600, batch_size=32, block_size=cfg.block_size,
             lr=3e-4, device=device, eval_every=300)
model.eval()
print("params:", cu.human(cu.count_params(model)))

# %% [markdown]
# ## 2 · Why naive generation is quadratic
#
# `TinyGPT.generate` re-runs the whole context for every new token. But the K and
# V vectors for tokens already processed **cannot change** — attention is causal,
# so nothing about the past depends on the future. Recomputing them is pure
# waste.
#
# Generating `n` tokens naively costs `O(n²)` forward work. With a cache it is
# `O(n)` forwards of a single token each.

# %%
print(f"{'tokens':>8} {'naive: positions processed':>28} {'cached':>10}")
for n in (10, 100, 1000, 4096):
    print(f"{n:>8} {n * (n + 1) // 2:>28,} {n:>10,}")
print("\nAt n=4096 the naive version does ~2000x more attention work.")

# %% [markdown]
# ## 3 · Implement the cache
#
# In attention, when generating we pass a **single** new token (`T=1`):
#
# 1. compute `q, k, v` for that one position
# 2. append `k, v` to the cache → full `(B, h, T_past+1, d)` tensors
# 3. attend the single query against all cached keys
#
# No causal mask is needed at all: a lone query at the newest position may attend
# to everything in the cache by construction. That is worth pausing on — the mask
# exists only because training processes all positions at once.

# %%
class CachedAttention(nn.Module):
    """Causal self-attention with an incremental KV cache."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.register_buffer("tril", torch.tril(
            torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, -1),
            persistent=False)

    def forward(self, x, cache=None, use_cache=False):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, h, d).transpose(1, 2)
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)      # (B, h, T_past+T, d)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v) if use_cache else None

        S = k.size(2)
        att = (q @ k.transpose(-2, -1)) * d ** -0.5          # (B, h, T, S)
        if T > 1:
            # prefill: standard causal mask over the last T rows of the S columns
            att = att.masked_fill(self.tril[:, :, S - T:S, :S] == 0, float("-inf"))
        # T == 1: the single query may see the whole cache, no mask required
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), new_cache


class CachedGPT(nn.Module):
    """Same weights as TinyGPT, but a forward pass that threads a KV cache."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(cfg.d_model),
                "attn": CachedAttention(cfg),
                "ln2": nn.LayerNorm(cfg.d_model),
                "ff": nn.ModuleDict({
                    "fc": nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias),
                    "proj": nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias),
                }),
            }))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    @classmethod
    def from_tinygpt(cls, src: TinyGPT):
        """Copy weights across — identical parameters, different plumbing."""
        m = cls(src.cfg)
        m.tok_emb.load_state_dict(src.tok_emb.state_dict())
        m.pos_emb.load_state_dict(src.pos_emb.state_dict())
        m.ln_f.load_state_dict(src.ln_f.state_dict())
        m.head.load_state_dict(src.head.state_dict())
        for dst, s in zip(m.blocks, src.blocks):
            dst["ln1"].load_state_dict(s.ln1.state_dict())
            dst["ln2"].load_state_dict(s.ln2.state_dict())
            dst["attn"].qkv.load_state_dict(s.attn.qkv.state_dict())
            dst["attn"].proj.load_state_dict(s.attn.proj.state_dict())
            dst["ff"]["fc"].load_state_dict(s.ff.fc.state_dict())
            dst["ff"]["proj"].load_state_dict(s.ff.proj.state_dict())
        return m

    def forward(self, idx, caches=None, use_cache=False, past_len=0):
        B, T = idx.shape
        # positions continue from where the cache left off
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            y, nc = block["attn"](block["ln1"](x), cache, use_cache)
            x = x + y
            h = block["ln2"](x)
            x = x + block["ff"]["proj"](F.gelu(block["ff"]["fc"](h)))
            new_caches.append(nc)
        return self.head(self.ln_f(x)), new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, greedy=False):
        self.eval()
        # --- prefill: process the whole prompt once, filling the cache ---
        logits, caches = self(idx, use_cache=True, past_len=0)
        past_len = idx.size(1)
        out = idx
        for _ in range(max_new_tokens):
            step_logits = logits[:, -1, :]
            if greedy:
                nxt = step_logits.argmax(-1, keepdim=True)
            else:
                filtered = cu.filter_logits(step_logits, temperature, top_k, top_p)
                nxt = torch.multinomial(F.softmax(filtered, dim=-1), 1)
            out = torch.cat([out, nxt], dim=1)
            if past_len >= self.cfg.block_size:
                break                            # positional table exhausted
            # --- decode: ONE token, reusing the cache ---
            logits, caches = self(nxt, caches=caches, use_cache=True,
                                  past_len=past_len)
            past_len += 1
        return out


cached = CachedGPT.from_tinygpt(model).to(device).eval()
print("weights copied:", cu.human(cu.count_params(cached)), "params")

# %% [markdown]
# ### Correctness first, speed second
#
# A cache bug produces *plausible* text, so it is easy to ship. The test is that
# greedy decoding must be **bit-identical** to the naive path.

# %%
prompt = "The king speaks"
idx = torch.tensor([tok.encode(prompt)], device=device)

with torch.no_grad():
    naive_logits, _ = model(idx)
    cached_logits, _ = cached(idx, use_cache=False)
print("prefill logits match:",
      torch.allclose(naive_logits, cached_logits, atol=1e-4))

naive_out = model.generate(idx, 40, temperature=0.01)          # ~greedy
cached_out = cached.generate(idx, 40, greedy=True)
print("\nnaive :", repr(tok.decode(naive_out[0])))
print("cached:", repr(tok.decode(cached_out[0])))

with torch.no_grad():
    step_naive, _ = model(cached_out[:, :10])
    step_cached, caches = cached(cached_out[:, :9], use_cache=True, past_len=0)
    step_cached, _ = cached(cached_out[:, 9:10], caches=caches, use_cache=True,
                            past_len=9)
print("\nincremental step == full recompute:",
      torch.allclose(step_naive[:, -1], step_cached[:, -1], atol=1e-4))

# %% [markdown]
# ### The speedup

# %%
def timed(fn, n=3):
    for _ in range(1):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


print(f"{'new tokens':>11} {'naive (s)':>11} {'cached (s)':>11} {'speedup':>9}")
for n_new in (32, 64, 110):
    t_naive = timed(lambda: model.generate(idx, n_new, temperature=0.01))
    t_cached = timed(lambda: cached.generate(idx, n_new, greedy=True))
    print(f"{n_new:>11} {t_naive:>11.3f} {t_cached:>11.3f} "
          f"{t_naive / t_cached:>8.2f}x")
print("""
Underwhelming? It should be, at this size. Each step here is so small that the
wall-clock is dominated by Python and kernel-launch overhead, which the cache
does nothing about -- both versions pay it once per token. The cache removes
*attention and matmul work*, which only dominates once the model and the context
are big enough. Two ways to see the real effect follow.""")

# %% [markdown]
# ### Effect 1: count the work exactly
#
# Wall-clock on a small model is a bad instrument — it measures Python overhead.
# Arithmetic is a perfect instrument. Below, the *exact* number of attention
# score entries and matmul FLOPs each approach performs to generate `n` tokens
# from a short prompt. This is hardware-independent and it is what the cache
# actually changes.

# %%
def generation_work(n_new, prompt_len, cfg, n_params):
    """Score entries and FLOPs for naive re-forward vs single-token decode."""
    naive_positions = sum(prompt_len + i for i in range(n_new))
    cached_positions = prompt_len + n_new - 1          # prefill + one per step

    def scores(total_positions_per_step, steps):
        return steps * total_positions_per_step

    naive_scores = sum(cfg.n_heads * cfg.n_layers * (prompt_len + i) ** 2
                       for i in range(n_new))
    cached_scores = (cfg.n_heads * cfg.n_layers * prompt_len ** 2
                     + sum(cfg.n_heads * cfg.n_layers * (prompt_len + i)
                           for i in range(n_new)))
    naive_flops = 2 * n_params * naive_positions
    cached_flops = 2 * n_params * cached_positions
    return naive_positions, cached_positions, naive_scores, cached_scores, \
        naive_flops, cached_flops


print(f"{'tokens':>7}{'positions (naive)':>19}{'(cached)':>10}"
      f"{'attn scores naive':>19}{'(cached)':>12}{'FLOP ratio':>12}")
for n_new in (32, 128, 512, 2048):
    np_, cp, ns, cs, nf, cf = generation_work(n_new, 8, cfg,
                                              cu.count_params(model))
    print(f"{n_new:>7}{np_:>19,}{cp:>10,}{ns:>19,}{cs:>12,}{nf / cf:>11.1f}x")

print("""
The cache is not a constant-factor optimization -- the FLOP ratio grows roughly
as n/2, so at 2048 generated tokens the naive path does ~1000x the model FLOPs
and ~1400x the attention work. Whether your stopwatch shows any of that depends
entirely on whether compute or overhead dominates on your hardware, which is the
next cell's point.""")

# %% [markdown]
# ### Effect 2: a longer generation on a bigger model

# %%
long_cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=512, d_model=384,
                     n_heads=6, n_layers=6, dropout=0.0, flash=False)
naive_big = TinyGPT(long_cfg).to(device).eval()     # untrained: timing, not text
cached_big = CachedGPT.from_tinygpt(naive_big).to(device).eval()

start_big = torch.randint(0, tok.vocab_size, (1, 8), device=device)
print("(untrained weights, 10x the parameters of the model above)")
print(f"{'new tokens':>11} {'naive (s)':>11} {'cached (s)':>11} {'speedup':>9}")
for n_new in (128, 256, 500):
    t_naive = timed(lambda: naive_big.generate(start_big, n_new, temperature=0.01), n=1)
    t_cached = timed(lambda: cached_big.generate(start_big, n_new, greedy=True), n=1)
    print(f"{n_new:>11} {t_naive:>11.2f} {t_cached:>11.2f} "
          f"{t_naive / t_cached:>8.2f}x")
print("""
The measured speedup grows with length, in the direction the theory predicts,
but it lands far below the FLOP ratio above. That gap is the lesson: on a small
model, a laptop GPU and short contexts, generation is bound by per-step launch
overhead that the cache cannot touch.

Re-run this notebook on a CUDA box with a larger model and you will see the two
numbers converge. Notebook 09's scaling reasoning applies here too -- always ask
whether the thing you are optimizing is the thing that is actually costing you.
Every production serving stack has a KV cache because at 7B parameters, 32
layers and 4k+ context, it IS the thing.""")

# %% [markdown]
# ## 4 · Prefill vs decode: two different problems
#
# **Prefill** processes the whole prompt in one forward pass: big matmuls, high
# arithmetic intensity, **compute bound**. Good GPU utilization.
#
# **Decode** processes one token at a time: every weight is loaded from memory to
# be used *once*, **memory-bandwidth bound**. Terrible utilization — which is why
# batching many users' decode steps together is the core trick of production
# serving (continuous batching), and why the KV cache size limits batch size.

# %%
n_params = cu.count_params(model)
bytes_per_param = 2                                    # bf16
weight_bytes = n_params * bytes_per_param
print(f"model weights: {weight_bytes / 1e6:.1f} MB")
print("\nEvery decode step must read all weights from memory. Achievable tokens/s")
print("for a single sequence is therefore roughly bandwidth / weight_bytes:\n")
for name, bw in (("RTX 4090", 1008e9), ("A100 80GB", 2039e9), ("H100 SXM", 3350e9)):
    print(f"  {name:<10} {bw / 1e9:>5.0f} GB/s -> "
          f"{bw / weight_bytes:>10,.0f} tok/s for this model, "
          f"{bw / (7e9 * 2):>7,.0f} tok/s for a 7B model")
print("\nThat 7B number is the theoretical ceiling for one sequence. Batching")
print("amortizes the weight reads across sequences — the only way past it.")

# %%
prompt_long = torch.tensor([tok.encode(text[:120])], device=device)
with torch.no_grad():
    t_prefill = timed(lambda: cached(prompt_long, use_cache=True, past_len=0), n=5)
    _, caches = cached(prompt_long, use_cache=True, past_len=0)
    one = prompt_long[:, -1:]
    t_decode = timed(lambda: cached(one, caches=caches, use_cache=True,
                                    past_len=120), n=20)
print(f"prefill 120 tokens : {t_prefill * 1e3:7.2f} ms "
      f"({120 / t_prefill:>9,.0f} tok/s)")
print(f"decode 1 token     : {t_decode * 1e3:7.2f} ms "
      f"({1 / t_decode:>9,.0f} tok/s)")
print(f"\nPrefill is ~{(120 / t_prefill) / (1 / t_decode):.0f}x more efficient "
      "per token. Same model, same weights.")

# %% [markdown]
# ## 5 · KV cache memory
#
# The cache is `2 · L · h_kv · d · T · B` values. It grows with **every** token
# and it is why long-context serving is expensive — and why GQA (notebook 08)
# matters so much.

# %%
def cache_gb(L, h_kv, d, T, B, bytes_per=2):
    return 2 * L * h_kv * d * T * B * bytes_per / 1e9


print("this model (L=6, h=6, d=32):")
for T_ in (128, 1024):
    print(f"  T={T_:>5}, batch 32: {cache_gb(6, 6, 32, T_, 32) * 1000:.2f} MB")
print("\nLlama-3-8B (L=32, 8 KV heads, d=128):")
for T_ in (2048, 8192, 32768):
    for B_ in (1, 16):
        print(f"  T={T_:>6}, batch {B_:>2}: {cache_gb(32, 8, 128, T_, B_):6.2f} GB")
print("\nA 32k context at batch 16 costs more memory than the weights themselves.")

# %% [markdown]
# ## 6 · Decoding strategies
#
# The model gives you a distribution; a *strategy* turns it into a token. This
# choice affects perceived quality more than another few hundred training steps.

# %%
def sample_with(strategy: dict, n_tokens=140, seed=0):
    torch.manual_seed(seed)
    out = cached.generate(idx, n_tokens, **strategy)
    return tok.decode(out[0])


strategies = {
    "greedy (argmax)": {"greedy": True},
    "temperature 0.3": {"temperature": 0.3},
    "temperature 1.0": {"temperature": 1.0},
    "temperature 1.6": {"temperature": 1.6},
    "top-k 10": {"temperature": 1.0, "top_k": 10},
    "top-p 0.9 (nucleus)": {"temperature": 1.0, "top_p": 0.9},
    "top-k 40 + top-p 0.95": {"temperature": 0.9, "top_k": 40, "top_p": 0.95},
}
for name, kw in strategies.items():
    print(f"--- {name} ---")
    print(repr(sample_with(kw)))
    print()

# %% [markdown]
# ### What each knob does to the distribution
#
# - **greedy** — always the argmax. Deterministic, and prone to loops, because a
#   repeated phrase is often locally the most likely continuation.
# - **temperature** — divide logits by `T`. Sharpen (<1) or flatten (>1).
# - **top-k** — keep the `k` best, renormalize. Fixed budget regardless of how
#   confident the model is.
# - **top-p / nucleus** — keep the smallest set whose mass exceeds `p`. *Adaptive*:
#   narrow when the model is confident, wide when it isn't. Usually the better
#   default.
# - **min-p** — keep tokens with probability above `p × p_max`. Newer, similarly
#   adaptive, robust at high temperature.

# %%
# A well-trained model is often extremely confident, and filtering a
# near-one-hot distribution shows nothing. So search a stretch of real text for
# the position where the model is LEAST sure — that is where the strategy
# actually decides something.
scan = torch.tensor([tok.encode(text[2000:2000 + cfg.block_size])], device=device)
with torch.no_grad():
    scan_logits, _ = cached(scan, use_cache=False)
p_all = F.softmax(scan_logits[0], dim=-1)                     # (T, V)
ents = -(p_all * p_all.clamp_min(1e-9).log()).sum(-1)
where = int(ents.argmax())
raw = scan_logits[0, where]
probs = F.softmax(raw, dim=-1)
ctx_str = tok.decode(scan[0, max(0, where - 20):where + 1])
print(f"least-confident position in the sample: after {ctx_str!r}")
print(f"entropy {ents[where]:.3f} nats "
      f"(max {math.log(tok.vocab_size):.3f}); the model's own prompt scored "
      f"{ents.min():.3f} at its most confident position)")
print(f"\n{'strategy':<22}{'candidates kept':>17}{'renormalized mass of top-1':>29}")
for name, kw in (("full distribution", {}), ("top-k 5", {"top_k": 5}),
                 ("top-k 20", {"top_k": 20}), ("top-p 0.9", {"top_p": 0.9}),
                 ("top-p 0.5", {"top_p": 0.5})):
    p = F.softmax(cu.filter_logits(raw.clone(), 1.0, **kw), dim=-1)
    print(f"{name:<22}{int((p > 0).sum()):>17}{p.max().item():>29.3f}")


def min_p_filter(logits, min_p=0.1):
    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < threshold, float("-inf"))


p = F.softmax(min_p_filter(raw.clone(), 0.1), dim=-1)
print(f"{'min-p 0.1':<22}{int((p > 0).sum()):>17}{p.max().item():>29.3f}")

# %% [markdown]
# ### Repetition penalty
#
# A cheap fix for loops: divide (or subtract from) the logits of tokens already
# generated. Useful for small models; large ones rarely need it, and it can
# damage legitimate repetition (names, code, tables).

# %%
def generate_with_penalty(m, idx, n_tokens, penalty=1.2, temperature=1.0):
    out = idx.clone()
    logits, caches = m(out, use_cache=True, past_len=0)
    past = out.size(1)
    for _ in range(n_tokens):
        step = logits[:, -1, :].clone()
        for t in set(out[0].tolist()):
            step[0, t] /= penalty if step[0, t] > 0 else 1 / penalty
        nxt = torch.multinomial(F.softmax(step / temperature, dim=-1), 1)
        out = torch.cat([out, nxt], dim=1)
        if past >= m.cfg.block_size:
            break
        logits, caches = m(nxt, caches=caches, use_cache=True, past_len=past)
        past += 1
    return out


torch.manual_seed(0)
print("no penalty  :", repr(tok.decode(cached.generate(idx, 100, temperature=1.0)[0])))
torch.manual_seed(0)
print("penalty 1.3 :", repr(tok.decode(generate_with_penalty(cached, idx, 100,
                                                             penalty=1.3)[0])))

# %% [markdown]
# ## 7 · Beam search
#
# Sampling picks one path; beam search keeps the `k` highest-*total-probability*
# sequences at every step. It finds higher-likelihood text, which is what you
# want for translation or transcription (there is a right answer) and *not* what
# you want for open-ended generation (high-likelihood text is bland and
# repetitive — the "likelihood trap").

# %%
@torch.no_grad()
def beam_search(m, idx, max_new_tokens, beams=4, length_penalty=1.0):
    logits, caches = m(idx, use_cache=True, past_len=0)
    logprobs = F.log_softmax(logits[:, -1], dim=-1)
    top = logprobs.topk(beams, dim=-1)

    seqs = torch.cat([idx.repeat(beams, 1), top.indices.view(beams, 1)], dim=1)
    scores = top.values.view(beams)
    caches = [(k.repeat(beams, 1, 1, 1), v.repeat(beams, 1, 1, 1))
              for k, v in caches]
    past = idx.size(1)

    for _ in range(max_new_tokens - 1):
        if past >= m.cfg.block_size - 1:
            break
        logits, caches = m(seqs[:, -1:], caches=caches, use_cache=True, past_len=past)
        past += 1
        lp = F.log_softmax(logits[:, -1], dim=-1)               # (beams, V)
        cand = scores.unsqueeze(1) + lp                          # total logprob
        flat = cand.view(-1)
        best = flat.topk(beams)
        beam_idx = best.indices // lp.size(-1)
        tok_idx = best.indices % lp.size(-1)
        seqs = torch.cat([seqs[beam_idx], tok_idx.unsqueeze(1)], dim=1)
        scores = best.values
        caches = [(k[beam_idx], v[beam_idx]) for k, v in caches]

    norm = scores / (seqs.size(1) ** length_penalty)
    order = norm.argsort(descending=True)
    return seqs[order], scores[order], norm[order]


seqs, scores, norm = beam_search(cached, idx, 60, beams=4)
print(f"{'total logprob':>14} {'per-token':>10}  text")
for s, total, per in zip(seqs, scores, norm):
    print(f"{total:>14.2f} {per:>10.3f}  {tok.decode(s)!r}")

greedy_seq = cached.generate(idx, 60, greedy=True)
with torch.no_grad():
    lg, _ = cached(greedy_seq[:, :-1], use_cache=False)
    lp = F.log_softmax(lg, -1).gather(2, greedy_seq[:, 1:].unsqueeze(-1)).sum()
print(f"\ngreedy total logprob: {lp.item():.2f} <= best beam {scores[0]:.2f}")
print("Beam search finds more probable text. Whether that is BETTER depends")
print("entirely on the task.")

# %% [markdown]
# ## 8 · Batched generation
#
# Generating 32 sequences at once costs barely more wall-clock than one, because
# decode is bandwidth bound: the weights get read once for the whole batch.

# %%
for B in (1, 8, 32):
    start = torch.zeros((B, 1), dtype=torch.long, device=device)
    t = timed(lambda: cached.generate(start, 60, temperature=0.9, top_k=20), n=2)
    print(f"batch {B:>3}: {t:.3f}s total -> {B * 60 / t:>8,.0f} tok/s "
          f"({t / (B * 60) * 1e3:.2f} ms per token)")
print("\nThroughput scales nearly linearly with batch size while per-sequence")
print("latency barely moves. Production servers exploit exactly this.")

# %% [markdown]
# ## 9 · Quantization, by hand
#
# Store weights in int8 instead of fp32: 4x smaller, and — because decode is
# bandwidth bound — roughly proportionally faster. Per-channel symmetric
# quantization is enough to understand the idea:
#
# ```
# scale = max(|W|) / 127        (per output channel)
# W_int8 = round(W / scale)
# W_dequant = W_int8 * scale
# ```

# %%
class Int8Linear(nn.Module):
    """Weight-only int8 quantization, dequantizing on the fly."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        W = lin.weight.data
        scale = W.abs().amax(dim=1, keepdim=True) / 127.0        # per out-channel
        self.register_buffer("qweight", torch.round(W / scale).to(torch.int8))
        self.register_buffer("scale", scale)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.clone())

    def forward(self, x):
        W = self.qweight.to(x.dtype) * self.scale
        return F.linear(x, W, self.bias)


lin = nn.Linear(256, 256)
q = Int8Linear(lin)
x = torch.randn(4, 256)
err = (lin(x) - q(x)).abs().mean() / lin(x).abs().mean()
print(f"relative output error: {err:.4%}")
print(f"weight bytes: fp32 {lin.weight.numel() * 4:,} -> "
      f"int8 {q.qweight.numel() + q.scale.numel() * 4:,}")

# %%
import copy


def quantize_model(m):
    m = copy.deepcopy(m)
    for name, module in m.named_modules():
        for cname, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child.out_features != m.cfg.vocab_size:
                setattr(module, cname, Int8Linear(child))
    return m


qmodel = quantize_model(model).to(device).eval()


def model_bytes(m):
    total = 0
    for p in m.parameters():
        total += p.numel() * p.element_size()
    for b in m.buffers():
        total += b.numel() * b.element_size()
    return total


fp32_loss = cu.estimate_loss(model, {"val": val_data}, 16, cfg.block_size,
                             device, iters=20)["val"]
int8_loss = cu.estimate_loss(qmodel, {"val": val_data}, 16, cfg.block_size,
                             device, iters=20)["val"]
print(f"{'precision':<10}{'size':>12}{'val loss':>11}")
print(f"{'fp32':<10}{model_bytes(model) / 1e6:>10.2f} MB{fp32_loss:>11.4f}")
print(f"{'int8':<10}{model_bytes(qmodel) / 1e6:>10.2f} MB{int8_loss:>11.4f}")
print(f"\nloss change: {int8_loss - fp32_loss:+.4f} for a "
      f"{model_bytes(model) / model_bytes(qmodel):.1f}x size reduction")

torch.manual_seed(0)
print("\nint8 sample:", repr(tok.decode(qmodel.generate(idx, 100, temperature=0.8,
                                                        top_k=20)[0])))

# %% [markdown]
# ### What real quantization adds
#
# - **Grouped scales** (e.g. one per 128 weights) instead of per-channel — much
#   lower error at 4 bits.
# - **Calibration** (GPTQ, AWQ): choose rounding to minimize error on real
#   activations, not just weight magnitude. This is what makes 4-bit viable.
# - **Fused int8/int4 kernels**: our version dequantizes to fp32 and gets no
#   speedup. Real kernels multiply in low precision.
# - **Outlier handling**: a few activation channels have huge magnitude and must
#   stay in higher precision (LLM.int8()).
#
# Practical rule of thumb: 8-bit is nearly free; 4-bit with a good method costs
# a little quality; below 4 bits, quality falls off quickly.

# %% [markdown]
# ## 10 · Speculative decoding, in one paragraph
#
# Since decode is bandwidth bound, one big model step costs about the same
# whether it verifies 1 token or 5. So: let a small **draft** model propose `k`
# tokens cheaply, then have the big model score all `k` in a *single* forward pass
# and accept the longest prefix that matches what it would have sampled. Output
# is distributionally identical to sampling from the big model alone, at 2-3x
# the speed when the draft agrees often.

# %%
print("""
draft (small):  the king spea|ks of the
verify (big) :  ^ ^   ^    ^  ^         one forward pass over all proposals
accept       :  the king spea      (first mismatch ends the run)

Cost per accepted token falls toward the draft model's cost. The technique is
free quality-wise, which is why it is now standard in serving stacks.
""")

# %% [markdown]
# ## Exercises
#
# 1. **Break the cache.** Remove `past_len` from the position calculation (always
#    start at 0) and compare greedy output against the naive path. Note how
#    plausible the wrong output looks — this is why the equality test matters.
# 2. **Sliding window.** Our cache stops at `block_size`. Implement a rolling
#    cache that drops the oldest entries, and explain why the output degrades
#    with learned absolute positions (and would not with ALiBi).
# 3. **Beam width.** Sweep beams = 1, 2, 4, 8, 16. Plot best total logprob and
#    subjective quality. Where does "more likely" stop meaning "better"?
# 4. **Group-wise int4.** Extend `Int8Linear` to 4 bits with a scale per group of
#    128 weights. Compare val loss against per-channel int8.
# 5. **Toy speculative decoding.** Train a 2-layer draft model, implement the
#    accept/reject loop against the 6-layer model, and measure the acceptance
#    rate and wall-clock speedup.
#
# ---
# **Next:** [13 · Interpretability and the capstone](13_interpretability_and_capstone.ipynb)
