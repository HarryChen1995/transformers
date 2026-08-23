# %% [markdown]
# # 08 · Modern architecture: RMSNorm, SwiGLU, GQA and flash attention
#
# **Goal:** upgrade the 2017/GPT-2 block to what 2023+ open models actually ship.
# Each change is small, we implement it, and we measure whether it earns its
# place — then assemble a LLaMA-style block and train it head-to-head against
# notebook 05's.
#
# **You will be able to:**
# 1. implement RMSNorm and say what it drops from LayerNorm and why
# 2. implement SwiGLU and keep the parameter count matched with the 2/3 rule
# 3. implement grouped-query attention and compute the KV-cache saving
# 4. use `F.scaled_dot_product_attention` and measure the memory it saves
# 5. read a modern model card (LLaMA, Qwen, Mistral) and recognize every part
#
# Prerequisites: notebooks 01–07.

# %%
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path.cwd()
if not (ROOT / "course_utils.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import course_utils as cu

cu.set_seed(1337)
device = cu.get_device()
print("device:", cu.describe_device(device))

# %% [markdown]
# ## 1 · What changed, and why
#
# | component | GPT-2 (2019) | LLaMA-style (2023+) | reason |
# |---|---|---|---|
# | norm | LayerNorm | **RMSNorm** | fewer ops, same quality |
# | norm placement | pre-norm | pre-norm (+ final norm) | unchanged, it was right |
# | activation | GELU | **SwiGLU** | better loss at matched params |
# | position | learned absolute | **RoPE** (nb 07) | relative, extrapolates |
# | attention | MHA | **GQA** | shrinks the KV cache |
# | attention kernel | manual softmax | **flash / SDPA** | O(T) memory, faster |
# | bias terms | everywhere | **none** | free parameters, no benefit |
#
# Nothing here is a paradigm shift. It is a decade of small wins compounding —
# which is itself the lesson.

# %% [markdown]
# ## 2 · RMSNorm
#
# LayerNorm centers (subtract the mean) and scales (divide by std), then applies
# a learned gain and bias. RMSNorm drops the centering and the bias:
#
# ```
# LayerNorm(x) = (x - mean) / sqrt(var + eps) * g + b
# RMSNorm(x)   =  x / sqrt(mean(x²) + eps) * g
# ```
#
# It turns out the re-centering was doing little work, and dropping it saves a
# pass over the data plus `C` parameters per norm.

# %%
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # rsqrt(mean(x^2)) computed in fp32 for stability, then cast back
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


x = torch.randn(2, 8, 64) * 3 + 1
ln, rms = nn.LayerNorm(64), RMSNorm(64)
print("input       rms:", x.pow(2).mean().sqrt().item())
print("LayerNorm   rms:", ln(x).pow(2).mean().sqrt().item(),
      "| mean:", ln(x).mean().item())
print("RMSNorm     rms:", rms(x).pow(2).mean().sqrt().item(),
      "| mean:", rms(x).mean().item(), "  <- NOT re-centered")
print("\nparams: LayerNorm", cu.count_params(ln), "| RMSNorm", cu.count_params(rms))

# %%
def bench(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3         # ms per call


big = torch.randn(32, 512, 768, device=device)
ln_d, rms_d = nn.LayerNorm(768).to(device), RMSNorm(768).to(device)
print(f"LayerNorm: {bench(lambda: ln_d(big)):.3f} ms")
print(f"RMSNorm  : {bench(lambda: rms_d(big)):.3f} ms")
print("(A few percent of total step time — small, free, and it compounds over"
      "\n 32-80 layers x 2 norms per layer.)")

# %% [markdown]
# ## 3 · SwiGLU
#
# The GPT-2 MLP is `down(GELU(up(x)))`. SwiGLU splits the up-projection in two
# and uses one half to **gate** the other:
#
# ```
# SwiGLU(x) = down( SiLU(gate(x)) * up(x) )
# ```
#
# That is three matrices instead of two, so to keep the parameter count matched
# you shrink the hidden size by 2/3: `4C -> 8C/3`, usually rounded to a multiple
# of 256. Every LLaMA-family model does exactly this.

# %%
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int | None = None,
                 multiple_of: int = 32, dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            hidden = int(2 * (4 * d_model) / 3)                 # the 2/3 rule
            hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden = hidden
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class GELUMLP(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        self.up = nn.Linear(d_model, mult * d_model, bias=False)
        self.down = nn.Linear(mult * d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))


C = 512
sw, ge = SwiGLU(C), GELUMLP(C)
print(f"SwiGLU  : hidden {sw.hidden:>5} | {cu.count_params(sw):>9,} params")
print(f"GELU MLP: hidden {4 * C:>5} | {cu.count_params(ge):>9,} params")
print(f"ratio: {cu.count_params(sw) / cu.count_params(ge):.3f} "
      "(the 2/3 rule keeps them comparable)")

# %% [markdown]
# ### Why gating helps
#
# `SiLU(gate(x)) * up(x)` is *multiplicative*: the network can suppress a
# feature entirely based on a different projection of the same input. A plain
# MLP can only apply a fixed pointwise nonlinearity. Multiplicative interactions
# are strictly more expressive per parameter — which is the empirical finding.

# %%
xs = torch.linspace(-3, 3, 7)
print("x         ", [f"{v:+.1f}" for v in xs.tolist()])
print("gelu(x)   ", [f"{v:+.2f}" for v in F.gelu(xs).tolist()])
print("silu(x)   ", [f"{v:+.2f}" for v in F.silu(xs).tolist()])
print("silu(x)*x ", [f"{v:+.2f}" for v in (F.silu(xs) * xs).tolist()],
      " <- a gate can flip sign and magnitude")

# %% [markdown]
# ## 4 · Grouped-query attention (GQA)
#
# At inference you cache K and V for every past token, at every layer. That cache
# is what limits how many users you can serve. GQA shrinks it: keep `n_heads`
# query heads, but only `n_kv_heads` key/value heads, and share each KV head
# across a group of query heads.
#
# - `n_kv_heads == n_heads` → standard MHA
# - `n_kv_heads == 1` → multi-query attention (MQA), maximum saving, some quality loss
# - `n_kv_heads == n_heads / 4..8` → GQA, the practical sweet spot
#
# First, see why anyone cares:

# %%
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per


print("Llama-3-8B shape (L=32, 32 query heads, head_dim=128), bf16, 8k context:")
for name, kv_heads in (("MHA (32 kv heads)", 32), ("GQA (8 kv heads)", 8),
                       ("MQA (1 kv head)", 1)):
    for batch in (1, 32):
        gb = kv_cache_bytes(32, kv_heads, 128, 8192, batch) / 1e9
        print(f"  {name:<20} batch {batch:>3}: {gb:>7.2f} GB of KV cache")
print("\nThat is why Llama-2-70B and Llama-3 both use 8 KV heads: a 4x cache")
print("reduction turns an infeasible batch size into a feasible one.")

# %%
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.0,
                 rope_base=10000.0, max_len=4096):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0
        self.n_rep = n_heads // self.n_kv_heads      # queries per KV head
        self.d = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.d, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d, d_model, bias=False)
        self.dropout_p = dropout

        inv = 1.0 / (rope_base ** (torch.arange(0, self.d, 2).float() / self.d))
        ang = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def _rope(self, t):
        T, d = t.size(-2), t.size(-1)
        cos = self.cos[:T].view(1, 1, T, d // 2)
        sin = self.sin[:T].view(1, 1, T, d // 2)
        t1, t2 = t[..., 0::2], t[..., 1::2]
        return torch.stack([t1 * cos - t2 * sin,
                            t1 * sin + t2 * cos], dim=-1).flatten(-2)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d).transpose(1, 2)

        q, k = self._rope(q), self._rope(k)

        # expand each KV head to cover its group of query heads.
        # repeat_interleave on the head dim: (B, kv, T, d) -> (B, h, T, d)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


for kv in (8, 2, 1):
    a = GroupedQueryAttention(512, 8, kv)
    print(f"n_kv_heads={kv}: params {cu.count_params(a):>8,} | "
          f"out {tuple(a(torch.randn(2, 16, 512)).shape)} | "
          f"KV cache is {8 // kv}x smaller than MHA")

# %% [markdown]
# Note what GQA does *not* change: the shape of the output, or the number of
# query heads. Only the K/V projections shrink. The expansion is free at
# training time (it's a view-like op) and at inference you store the *unexpanded*
# K/V, which is where the saving comes from.

# %% [markdown]
# ## 5 · Flash attention, via `scaled_dot_product_attention`
#
# The naive implementation materializes a `(B, h, T, T)` score matrix. Flash
# attention computes the same result in tiles, never storing the full matrix, so
# memory goes from `O(T²)` to `O(T)` — and it's *faster*, because attention is
# memory-bandwidth bound, not compute bound.
#
# PyTorch exposes it as `F.scaled_dot_product_attention`, which dispatches to
# the best available backend (flash / memory-efficient / math).

# %%
B, h, T, d = 8, 8, 512, 64
q, k, v = (torch.randn(B, h, T, d, device=device) for _ in range(3))


def manual_attention(q, k, v):
    T = q.size(-2)
    att = (q @ k.transpose(-2, -1)) * q.size(-1) ** -0.5
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    att = att.masked_fill(~mask, float("-inf"))
    return F.softmax(att, dim=-1) @ v


out_manual = manual_attention(q, k, v)
out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("same result:", torch.allclose(out_manual, out_sdpa, atol=1e-4))
print(f"\nmanual: {bench(lambda: manual_attention(q, k, v), n=30):.2f} ms")
print(f"sdpa  : {bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), n=30):.2f} ms")

score_matrix_mb = B * h * T * T * 4 / 1e6
print(f"\nscore matrix the manual version must store: {score_matrix_mb:.0f} MB")
print("sdpa stores none of it. At T=8192 that difference is the whole ballgame.")

# %%
if device.type == "cuda":
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH):
        try:
            with sdpa_kernel(backend):
                ms = bench(lambda: F.scaled_dot_product_attention(
                    q, k, v, is_causal=True), n=20)
            print(f"  {backend.name:<22} {ms:.2f} ms")
        except Exception as exc:
            print(f"  {backend.name:<22} unavailable ({type(exc).__name__})")
else:
    print("Backend selection is CUDA-specific — run this cell on the GPU box to")
    print("see FLASH_ATTENTION vs EFFICIENT_ATTENTION vs MATH side by side.")

# %% [markdown]
# ### The catch
#
# Fused attention gives you no access to the attention weights. If you want to
# visualize or interpret them (notebook 13), keep a manual path behind a flag —
# which is exactly what `tinygpt.py` does with `cfg.flash` and `return_weights`.

# %% [markdown]
# ## 6 · Assemble a LLaMA-style block

# %%
@dataclass
class ModernConfig:
    vocab_size: int = 47
    block_size: int = 128
    d_model: int = 192
    n_heads: int = 6
    n_kv_heads: int = 2
    n_layers: int = 6
    dropout: float = 0.1
    rope_base: float = 10000.0


class ModernBlock(nn.Module):
    def __init__(self, cfg: ModernConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                                          cfg.dropout, cfg.rope_base,
                                          max_len=cfg.block_size * 4)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, dropout=cfg.dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ModernGPT(nn.Module):
    """RMSNorm + SwiGLU + GQA + RoPE + no biases. A miniature LLaMA."""

    def __init__(self, cfg: ModernConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([ModernBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        x = self.drop(self.tok_emb(idx))          # no positional embedding: RoPE
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = cu.filter_logits(logits[:, -1], temperature, top_k)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


mcfg = ModernConfig()
m = ModernGPT(mcfg)
print(m.__class__.__name__, "|", cu.human(cu.count_params(m)), "params")
print("bias parameters:", sum(1 for n, _ in m.named_parameters() if "bias" in n))
print("out:", tuple(m(torch.randint(0, 47, (2, 32)))[0].shape))

# %% [markdown]
# ## 7 · Head-to-head at matched parameters
#
# The honest comparison: same corpus, same steps, same LR, similar parameter
# count. At this scale expect a modest win — the modern components were tuned at
# billions of parameters and trillions of tokens, and small models are noisy.
# The point is that *nothing regresses* while inference gets much cheaper.

# %%
from tinygpt import GPTConfig, TinyGPT

text = cu.get_text(verbose=False)
tok = cu.CharTokenizer(text)
train_data, val_data = cu.train_val_split(tok.encode(text))
splits = {"train": train_data, "val": val_data}
BS, STEPS = 64, 700

cu.set_seed(1337)
gpt2ish = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=BS,
                            d_model=192, n_heads=6, n_layers=6, dropout=0.1))
print(f"=== GPT-2 style: {cu.human(cu.count_params(gpt2ish))} params ===")
h_gpt = cu.train(gpt2ish, splits, steps=STEPS, batch_size=32, block_size=BS,
                 lr=1e-3, device=device, eval_every=350)

cu.set_seed(1337)
modern = ModernGPT(ModernConfig(vocab_size=tok.vocab_size, block_size=BS,
                                d_model=192, n_heads=6, n_kv_heads=2,
                                n_layers=6, dropout=0.1))
print(f"\n=== modern style: {cu.human(cu.count_params(modern))} params ===")
h_mod = cu.train(modern, splits, steps=STEPS, batch_size=32, block_size=BS,
                 lr=1e-3, device=device, eval_every=350)

# %%
print(f"{'model':<14}{'params':>10}{'val loss':>11}{'KV heads':>10}")
print(f"{'GPT-2 style':<14}{cu.human(cu.count_params(gpt2ish)):>10}"
      f"{h_gpt['val'][-1]:>11.4f}{6:>10}")
print(f"{'modern':<14}{cu.human(cu.count_params(modern)):>10}"
      f"{h_mod['val'][-1]:>11.4f}{2:>10}")
print(f"\nKV cache per token per layer: "
      f"{2 * 6 * 32} floats -> {2 * 2 * 32} floats (3x smaller)")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(h_gpt["step"], h_gpt["val"], "o-", label="GPT-2 style")
    ax.plot(h_mod["step"], h_mod["val"], "s-", label="modern (RMSNorm+SwiGLU+GQA+RoPE)")
    ax.set_xlabel("step"); ax.set_ylabel("val loss"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %%
prompt = "The king"
idx = torch.tensor([tok.encode(prompt)], device=device)
print(tok.decode(modern.generate(idx, 200, temperature=0.8, top_k=20)[0]))

# %% [markdown]
# ## 8 · Two more things you will meet in real models
#
# **QK-norm.** Normalize Q and K before the dot product (`RMSNorm` on the head
# dimension). It prevents attention logits from growing without bound during
# long training runs — a cheap stability fix used by Gemma, Qwen and others.
#
# **Mixture of experts (MoE).** Replace the single MLP with `N` MLPs and a router
# that sends each token to the top-`k` of them. Parameters grow `N`x while FLOPs
# per token stay flat, which is how Mixtral and DeepSeek-V3 get large capacity
# cheaply. The hard parts are load balancing and communication, not the math.

# %%
class QKNormAttention(nn.Module):
    """Minimal illustration of QK-norm inside attention."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.d), RMSNorm(self.d)

    def forward(self, x):
        B, T, Cx = x.shape
        q, k, v = self.qkv(x).split(Cx, dim=2)
        shape = lambda t: t.view(B, T, self.h, self.d).transpose(1, 2)
        q, k, v = self.q_norm(shape(q)), self.k_norm(shape(k)), shape(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, Cx))


qk = QKNormAttention(192, 6)
print("QK-norm attention out:", tuple(qk(torch.randn(2, 16, 192)).shape))


class TinyMoE(nn.Module):
    """Top-k routed mixture of experts, in the smallest honest form."""

    def __init__(self, d_model, n_experts=4, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model) for _ in range(n_experts)])

    def forward(self, x):
        B, T, Cx = x.shape
        flat = x.view(-1, Cx)                                   # (N, C)
        weights = F.softmax(self.router(flat), dim=-1)          # (N, E)
        topw, topi = weights.topk(self.top_k, dim=-1)           # (N, k)
        topw = topw / topw.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            hit = (topi == e)                                   # (N, k)
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            scale = (topw * hit).sum(-1)[rows].unsqueeze(-1)
            out[rows] += scale * expert(flat[rows])
        return out.view(B, T, Cx)


moe = TinyMoE(192, n_experts=4, top_k=2)
dense = SwiGLU(192)
print(f"\nMoE (4 experts, top-2): {cu.count_params(moe):,} params, "
      f"but only ~{cu.count_params(dense) * 2:,} used per token")
print("dense equivalent      :", f"{cu.count_params(dense):,} params, all used")
print("out:", tuple(moe(torch.randn(2, 8, 192)).shape))

# %% [markdown]
# ## Exercises
#
# 1. **Ablate one at a time.** Starting from `ModernGPT`, swap back to LayerNorm,
#    then to a GELU MLP, then to MHA, then to learned positions. Which single
#    change costs the most val loss at this scale?
# 2. **MQA quality cost.** Train `n_kv_heads` = 6, 3, 2, 1 and plot val loss
#    against KV-cache size. Where is the knee?
# 3. **The 2/3 rule.** Train SwiGLU with hidden = `4C` (unmatched) vs `8C/3`
#    (matched). How much of SwiGLU's win is architecture and how much is just
#    extra parameters?
# 4. **Memory measurement.** On CUDA, use `torch.cuda.max_memory_allocated()` to
#    measure peak memory for manual vs SDPA attention at T = 512, 1024, 2048.
#    Plot it and confirm the O(T²) vs O(T) scaling.
# 5. **Load balancing.** Add the standard auxiliary loss to `TinyMoE` (encourage
#    uniform expert usage) and log how often each expert fires with and without
#    it. Expect collapse onto one or two experts without the aux loss.
#
# ---
# **Next:** [09 · Training at scale](09_training_at_scale.ipynb) — LR schedules,
# mixed precision, gradient accumulation, `torch.compile`, and the GPU workflow.
