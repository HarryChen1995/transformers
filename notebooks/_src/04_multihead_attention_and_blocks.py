# %% [markdown]
# # 04 · Multi-head attention and the transformer block
#
# **Goal:** take the single head from notebook 03 and build the repeating unit of
# every transformer: multi-head attention + feed-forward, wrapped in residual
# connections and normalization.
#
# **You will be able to:**
# 1. implement multi-head attention with a single fused QKV projection
# 2. explain what multiple heads buy you that one wide head does not
# 3. say what the feed-forward network is for, and why it expands by 4x
# 4. explain residual connections as a "residual stream"
# 5. justify pre-norm over post-norm, and know where dropout goes
#
# Prerequisites: notebooks 01–03.

# %%
import math
import sys
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
# ## 1 · Why more than one head?
#
# A single softmax over positions must commit to *one* notion of relevance. But
# a token typically needs several unrelated things at once: the subject of its
# verb, the opening quote it must close, the last newline for line length.
#
# Multi-head attention runs `h` attentions in parallel on `d = C/h` dimensional
# slices, then concatenates. Same total FLOPs and parameters as one wide head —
# but `h` independent attention patterns instead of one.
#
# **Config first.** A dataclass keeps the shape letters in one place.

# %%
@dataclass
class Config:
    vocab_size: int = 47
    block_size: int = 64        # max T
    d_model: int = 128          # C
    n_heads: int = 4            # h
    n_layers: int = 4           # L
    dropout: float = 0.1
    bias: bool = True

    @property
    def head_dim(self) -> int:  # d
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        return self.d_model // self.n_heads


cfg = Config()
print(cfg, "\nhead_dim d =", cfg.head_dim)

# %% [markdown]
# ## 2 · The naive version: a list of heads
#
# Literally what notebook 03's exercise 5 asked for. Correct, and a useful
# reference — but it launches `h` separate small matmuls per projection, which
# wastes GPU throughput.

# %%
class Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.head_dim
        self.q = nn.Linear(cfg.d_model, d, bias=False)
        self.k = nn.Linear(cfg.d_model, d, bias=False)
        self.v = nn.Linear(cfg.d_model, d, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size,
                                                           cfg.block_size)))
        self.scale = d ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        att = (self.q(x) @ self.k(x).transpose(-2, -1)) * self.scale   # (B,T,T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return F.softmax(att, dim=-1) @ self.v(x)                      # (B,T,d)


class MultiHeadNaive(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = nn.ModuleList([Head(cfg) for _ in range(cfg.n_heads)])
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # each head: (B, T, d); concat along channels -> (B, T, h*d) = (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))


mha_naive = MultiHeadNaive(cfg)
x = torch.randn(2, 16, cfg.d_model)
print("out:", tuple(mha_naive(x).shape), "| params:",
      cu.human(cu.count_params(mha_naive)))

# %% [markdown]
# ## 3 · The fused version
#
# Three ideas, all mechanical:
#
# 1. **One `Linear(C, 3C)`** produces Q, K and V for *all* heads in one matmul.
# 2. **`view` + `transpose`** reinterpret `(B, T, C)` as `(B, h, T, d)`, so the
#    batched matmul does all heads at once. No data is copied.
# 3. **`transpose` + `contiguous` + `view`** merges heads back to `(B, T, C)`.
#
# Follow the shapes in the comments; this reshape dance is the single most
# important thing to internalize in this notebook.

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, causal: bool = True):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.causal = causal
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                 .view(1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x, return_weights: bool = False):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim

        # (B, T, C) -> (B, T, 3C) -> three (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, T, h, d) -> (B, h, T, d)
        q = q.view(B, T, h, d).transpose(1, 2)
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)

        # (B, h, T, d) @ (B, h, d, T) -> (B, h, T, T)
        att = (q @ k.transpose(-2, -1)) * d ** -0.5
        if self.causal:
            att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = self.attn_drop(att) @ v                       # (B, h, T, d)

        # merge heads: (B, h, T, d) -> (B, T, h, d) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))                 # (B, T, C)
        return (y, att) if return_weights else y


mha = MultiHeadAttention(cfg)
y, att = mha(x, return_weights=True)
print("out:", tuple(y.shape), "| att:", tuple(att.shape),
      "  <- (B, h, T, T): one attention matrix per head")
print("params:", cu.human(cu.count_params(mha)),
      "(same as naive, up to the qkv bias)")

# %% [markdown]
# ### Proving the reshape is a no-op
#
# The split-into-heads / merge-back-together pair must be an exact inverse.
# If you ever get this wrong (e.g. `view(B, h, T, d)` instead of
# `view(B, T, h, d).transpose(1, 2)`), the model still trains — just badly,
# because channels get scrambled across positions. Assert it once and move on.

# %%
B, T, C, h = 2, 5, 12, 3
d = C // h
z = torch.randn(B, T, C)
split = z.view(B, T, h, d).transpose(1, 2)                     # (B, h, T, d)
merged = split.transpose(1, 2).contiguous().view(B, T, C)      # (B, T, C)
print("round trip exact:", torch.equal(z, merged))

wrong = z.view(B, h, T, d)      # silently "works", scrambles the data
print("the wrong reshape gives the same SHAPE:", tuple(wrong.shape),
      "-> shape checks alone won't save you")

# %% [markdown]
# ### Do the heads actually differ?
#
# Nothing in the loss explicitly forces heads apart — they diverge because they
# start from different random init and specialize. Let's measure how different
# their attention patterns are (mean absolute difference between head matrices).

# %%
with torch.no_grad():
    _, att = mha(torch.randn(1, 12, cfg.d_model), return_weights=True)
a = att[0]                                                     # (h, T, T)
print("pairwise mean |difference| between untrained heads:")
for i in range(cfg.n_heads):
    for j in range(i + 1, cfg.n_heads):
        print(f"  head {i} vs {j}: {(a[i] - a[j]).abs().mean():.4f}")
print("\n(re-run this at the end of notebook 05 on a TRAINED model — the spread grows)")

# %% [markdown]
# ## 4 · The feed-forward network
#
# Attention **moves information between positions**. It does not do much
# *computation* — each output is a linear combination of value vectors. The
# per-position MLP is where the nonlinear work happens: every token, independently,
# gets pushed through a wide hidden layer.
#
# The 4x expansion ratio is empirical convention from the original paper, kept by
# GPT-2/3. Notebook 08 replaces GELU with SwiGLU and adjusts the ratio to keep
# the parameter count matched.

# %%
class FeedForward(nn.Module):
    def __init__(self, cfg, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, mult * cfg.d_model, bias=cfg.bias),  # C -> 4C
            nn.GELU(),
            nn.Linear(mult * cfg.d_model, cfg.d_model, bias=cfg.bias),  # 4C -> C
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)                                    # (B,T,C)->(B,T,C)


ff = FeedForward(cfg)
print("out:", tuple(ff(x).shape), "| params:", cu.human(cu.count_params(ff)))
print(f"\nattention params : {cu.count_params(mha):>8,}")
print(f"mlp params       : {cu.count_params(ff):>8,}")
print("=> roughly 2/3 of a transformer's parameters live in the MLPs")

# %% [markdown]
# ### Why a nonlinearity at all?
#
# Without it, `W2 @ (W1 @ x)` collapses into a single linear map and the extra
# layer buys nothing. GELU is a smooth gate: it passes large positive values
# through and softly suppresses negatives.

# %%
xs = torch.linspace(-3, 3, 7)
print("x    ", xs.tolist())
print("relu ", F.relu(xs).round(decimals=2).tolist())
print("gelu ", F.gelu(xs).round(decimals=2).tolist())

try:
    import matplotlib.pyplot as plt
    grid = torch.linspace(-4, 4, 200)
    plt.figure(figsize=(5, 3))
    for fn, name in ((F.relu, "ReLU"), (F.gelu, "GELU"), (F.silu, "SiLU/Swish")):
        plt.plot(grid, fn(grid), label=name)
    plt.legend(); plt.grid(alpha=0.3); plt.title("activation functions")
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %% [markdown]
# ## 5 · Residual connections: the residual stream
#
# `x = x + sublayer(x)` is doing more than "helping gradients flow". The right
# mental model is a **residual stream**: a `(B, T, C)` bus running the depth of
# the network that every block *reads from* and *writes into* additively. Blocks
# communicate through it rather than transforming a hidden state in place.
#
# Mechanically, the `+` gives gradients a path that skips every sublayer, so a
# 48-layer stack still trains.

# %%
def grad_norm_through_depth(residual: bool, depth: int = 24, C: int = 64):
    """Gradient magnitude reaching the input, with and without residuals."""
    torch.manual_seed(0)
    layers = [nn.Sequential(nn.Linear(C, C), nn.GELU()) for _ in range(depth)]
    inp = torch.randn(1, C, requires_grad=True)
    z = inp
    for layer in layers:
        z = z + layer(z) if residual else layer(z)
    z.sum().backward()
    return inp.grad.norm().item()


print(f"depth 24, no residual : grad norm at input = {grad_norm_through_depth(False):.3e}")
print(f"depth 24, residual    : grad norm at input = {grad_norm_through_depth(True):.3e}")
print("\nWithout the skip path the signal decays by orders of magnitude — the")
print("early layers barely learn. This is what residuals fix.")

# %% [markdown]
# ## 6 · Normalization, and where to put it
#
# `LayerNorm` normalizes **across the channel dimension**, per token: subtract
# the mean, divide by the standard deviation, then apply a learned scale and
# shift. Unlike BatchNorm it does not mix information across the batch or across
# positions, which is exactly what a sequence model needs.

# %%
ln = nn.LayerNorm(8)
sample = torch.randn(2, 3, 8) * 5 + 10          # badly scaled input
out = ln(sample)
print("input  mean/std per token:", sample.mean(-1).flatten().round(decimals=2).tolist(),
      "/", sample.std(-1).flatten().round(decimals=2).tolist())
print("output mean/std per token:", out.mean(-1).flatten().round(decimals=2).tolist(),
      "/", out.std(-1).flatten().round(decimals=2).tolist())
print("learnable params:", [n for n, _ in ln.named_parameters()])

# %% [markdown]
# ### Post-norm (2017) vs pre-norm (everything since)
#
# ```
# post-norm:  x = LayerNorm(x + attn(x))       # original paper
# pre-norm:   x = x + attn(LayerNorm(x))       # GPT-2 onward
# ```
#
# Pre-norm keeps the residual path *clean* — an unnormalized identity from input
# to output — so deep stacks train without learning-rate warmup gymnastics. In
# post-norm, every block re-normalizes the stream, which repeatedly rescales the
# gradient path. Use pre-norm.

# %%
class Block(nn.Module):
    """One pre-norm transformer decoder block. (B, T, C) -> (B, T, C)."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # communicate across positions
        x = x + self.ff(self.ln2(x))       # compute within each position
        return x


block = Block(cfg)
print("block out:", tuple(block(x).shape), "| params:",
      cu.human(cu.count_params(block)))

# %% [markdown]
# ### Measuring the difference

# %%
class PostNormBlock(Block):
    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x


for name, cls in (("pre-norm", Block), ("post-norm", PostNormBlock)):
    torch.manual_seed(0)
    stack = nn.Sequential(*[cls(cfg) for _ in range(12)])
    inp = torch.randn(1, 16, cfg.d_model, requires_grad=True)
    stack(inp).pow(2).mean().backward()
    print(f"{name:>10}: 12-block stack, grad norm at input = {inp.grad.norm():.4f}")

# %% [markdown]
# ## 7 · Dropout: three placements, one rate
#
# In the block above dropout appears in three places, each with a distinct job:
#
# 1. **attention dropout** — on the softmax weights: randomly ignore some
#    connections, so no single one is load-bearing.
# 2. **residual dropout** — on each sublayer's output before it is added back.
# 3. **embedding dropout** — on token+position embeddings (notebook 05).
#
# On a small corpus like ours, dropout is doing real work; at internet scale it
# is often set to 0 because there is no time to overfit.

# %%
print("dropout locations:")
for n, m in block.named_modules():
    if isinstance(m, nn.Dropout):
        print(f"  {n:<24} p={m.p}")

# %% [markdown]
# ## 8 · Stack blocks and count parameters
#
# Where the parameters actually live, for a real-ish config. Attention is
# `4C²` per layer (Q, K, V, output projection); the MLP is `8C²` (up 4x, down 4x).
# Hence "2/3 of the parameters are MLP".

# %%
def param_report(cfg):
    C, L, V = cfg.d_model, cfg.n_layers, cfg.vocab_size
    attn = 4 * C * C * L
    mlp = 8 * C * C * L
    emb = V * C + cfg.block_size * C
    head = C * V
    total = attn + mlp + emb + head
    rows = [("attention (4C^2 * L)", attn), ("mlp (8C^2 * L)", mlp),
            ("embeddings", emb), ("output head", head)]
    for name, n in rows:
        print(f"  {name:<24} {n:>12,}  ({100 * n / total:4.1f}%)")
    print(f"  {'TOTAL':<24} {total:>12,}  ({cu.human(total)})")


print("this notebook's toy config:")
param_report(cfg)
print("\nGPT-2 small (C=768, L=12, V=50257, block=1024):")
param_report(Config(vocab_size=50257, block_size=1024, d_model=768,
                    n_heads=12, n_layers=12))

# %%
stack = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
out = stack(x)
print("\n4 blocks:", tuple(x.shape), "->", tuple(out.shape),
      "  <- shape is invariant, which is why blocks are stackable")

# %% [markdown]
# ## 9 · Sanity check: does a block-stack actually help?
#
# Quick head-to-head against notebook 03's single head, same budget of steps.
# (No positional embeddings yet — that is notebook 05, and it will help again.)

# %%
text = cu.get_text(verbose=False)
tok = cu.CharTokenizer(text)
train_data, val_data = cu.train_val_split(tok.encode(text))
splits = {"train": train_data, "val": val_data}


class BlockStackLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.blocks(self.tok_emb(idx))
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss


cu.set_seed(1337)
small = Config(vocab_size=tok.vocab_size, block_size=64, d_model=128,
               n_heads=4, n_layers=4, dropout=0.1)
model = BlockStackLM(small)
print("params:", cu.human(cu.count_params(model)))
hist = cu.train(model, splits, steps=800, batch_size=32, block_size=64,
                lr=1e-3, device=device, eval_every=200)
print("\nCompare against the same-corpus baselines you trained earlier:")
print("  nb 02 bigram      (2.2K params)")
print("  nb 03 single head (18K params)")
print("  this stack        (805K params)")
print("\nAbsolute loss values depend entirely on your corpus (tiny-shakespeare is")
print("much harder than the synthetic fallback). What must hold is the ORDERING.")

# %% [markdown]
# ## Exercises
#
# 1. **Head count sweep.** Fix `d_model=128` and train with `n_heads` = 1, 2, 4,
#    8, 16 for 800 steps each. Plot final val loss vs head count. Where does it
#    stop helping, and why does `head_dim` getting tiny hurt?
# 2. **Scramble the heads.** Replace the correct reshape with
#    `q.view(B, h, T, d)` and retrain. How much loss does the bug cost? (Lesson:
#    silent shape bugs cost accuracy, not exceptions.)
# 3. **Kill the MLP.** Remove the feed-forward sublayer and add layers until the
#    parameter count matches. Which spends the budget better?
# 4. **Post-norm at depth.** Train `n_layers=12` with post-norm vs pre-norm at
#    `lr=3e-3`. Post-norm should be unstable or diverge.
# 5. **Expansion ratio.** Try `mult` = 1, 2, 4, 8 in `FeedForward`. Plot val loss
#    against parameter count — is 4x actually the sweet spot at this scale?
#
# ---
# **Next:** [05 · Build and train TinyGPT end to end](05_build_and_train_tinygpt.ipynb) —
# add positional embeddings, weight tying, proper init and generation.
