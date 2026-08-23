# %% [markdown]
# # 07 · Positional encodings: sinusoidal, learned, RoPE and ALiBi
#
# **Goal:** understand every positional scheme in common use, implement each one,
# and test the property that actually matters in practice — whether the model
# still works on sequences longer than it was trained on.
#
# **You will be able to:**
# 1. derive the sinusoidal encoding and explain the frequency ladder
# 2. state the trade-off between learned absolute and relative schemes
# 3. implement rotary embeddings (RoPE) and verify their defining property
# 4. implement ALiBi in three lines and explain why it extrapolates
# 5. run a length-extrapolation experiment and read the result
#
# Prerequisites: notebooks 01–05.

# %%
import math
import sys
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
torch.set_printoptions(precision=3, sci_mode=False, linewidth=110)

# %% [markdown]
# ## 1 · Four schemes, one table
#
# | scheme | added where | absolute or relative | extrapolates? | used by |
# |---|---|---|---|---|
# | learned absolute | to embeddings | absolute | no (hard limit) | GPT-2, BERT, ViT |
# | sinusoidal | to embeddings | absolute (relative-ish) | poorly | original paper |
# | **RoPE** | rotates Q and K | relative | reasonably (+ tricks) | LLaMA, Qwen, Mistral |
# | **ALiBi** | bias on scores | relative | very well | BLOOM, MPT |
#
# The interesting axis is *absolute vs relative*. Attention fundamentally cares
# about the **distance** between two tokens, not their index in the file. Schemes
# that encode distance directly generalize better.

# %% [markdown]
# ## 2 · Learned absolute (notebook 05's approach)
#
# A `(block_size, C)` table, indexed by position, added to the token embedding.
# Maximum flexibility, zero structure — and a hard wall at `block_size`, because
# there is no row 129 to look up.

# %%
block_size, C = 128, 64
learned = nn.Embedding(block_size, C)
print("table:", tuple(learned.weight.shape))
try:
    learned(torch.tensor([block_size + 5]))
except Exception as exc:
    print("position 133 ->", type(exc).__name__, ": no row exists. Hard limit.")

# %% [markdown]
# ## 3 · Sinusoidal (Vaswani et al., 2017)
#
# Fixed, not learned. Dimension `2i` of position `pos` is
# `sin(pos / 10000^(2i/C))`, and dimension `2i+1` is the matching cosine.
#
# Read it as a **binary counter made of dials**: low dimensions spin fast (they
# encode fine position), high dimensions spin slowly (coarse position). Together
# they give every position a unique fingerprint, unbounded in length.

# %%
def sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1).float()               # (T, 1)
    i = torch.arange(0, d_model, 2).float()                        # (C/2,)
    div = torch.exp(-math.log(10000.0) * i / d_model)              # (C/2,)
    pe = torch.zeros(max_len, d_model)                             # (T, C)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


pe = sinusoidal_encoding(200, 64)
print("shape:", tuple(pe.shape), "| range:", (pe.min().item(), pe.max().item()))
print("\nwavelengths, from fast to slow dials:")
for dim in (0, 8, 16, 32, 62):
    wl = 2 * math.pi * 10000 ** (2 * (dim // 2) / 64)
    print(f"  dim {dim:>2}: wavelength {wl:>12,.1f} positions")

# %%
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    im = axes[0].imshow(pe.T, aspect="auto", cmap="RdBu")
    axes[0].set_title("sinusoidal encoding"); axes[0].set_xlabel("position")
    axes[0].set_ylabel("channel"); fig.colorbar(im, ax=axes[0], fraction=0.046)

    for dim in (0, 4, 16, 40):
        axes[1].plot(pe[:, dim], label=f"dim {dim}")
    axes[1].legend(fontsize=8); axes[1].set_title("a few channels")
    axes[1].set_xlabel("position"); axes[1].grid(alpha=0.3)

    sim = F.normalize(pe, dim=-1) @ F.normalize(pe, dim=-1).T
    im2 = axes[2].imshow(sim, cmap="viridis")
    axes[2].set_title("cosine similarity between positions")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %% [markdown]
# ### The relative-ness property
#
# The similarity plot has diagonal band structure: `pe[i] · pe[j]` depends
# mostly on `i - j`. That's the appealing part. The catch is that the encoding is
# *added to the embedding* and then mangled by learned projections, so the model
# has to work to recover distance — and it doesn't extrapolate well in practice.

# %%
sim = F.normalize(pe, dim=-1) @ F.normalize(pe, dim=-1).T
print("similarity as a function of distance (averaged over positions 50-150):")
for dist in (0, 1, 2, 5, 10, 25, 50):
    vals = [sim[i, i + dist].item() for i in range(50, 150)]
    print(f"  distance {dist:>3}: {sum(vals) / len(vals):+.3f}")

# %% [markdown]
# ## 4 · RoPE — rotary position embeddings
#
# The idea that won. Instead of *adding* anything, **rotate** each 2-dimensional
# slice of Q and K by an angle proportional to its position:
#
# ```
# q_m -> R(m·θ) q_m        k_n -> R(n·θ) k_n
# ```
#
# Because rotations compose, the dot product becomes
#
# ```
# (R(mθ) q) · (R(nθ) k) = q^T R((m-n)θ) k
# ```
#
# which depends **only on `m - n`**. Relative position falls out of the algebra —
# no extra parameters, no table, no length limit built in.

# %%
def rope_freqs(head_dim: int, max_len: int, base: float = 10000.0):
    """Precompute cos/sin for each (position, dimension-pair)."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, inv_freq)                  # (T, d/2)
    return angles.cos(), angles.sin()                  # each (T, d/2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate pairs of channels. x: (B, h, T, d) -> same shape."""
    T, d = x.size(-2), x.size(-1)
    cos = cos[:T].view(1, 1, T, d // 2)
    sin = sin[:T].view(1, 1, T, d // 2)
    x1, x2 = x[..., 0::2], x[..., 1::2]                # even / odd channels
    # standard 2-D rotation applied to each (x1, x2) pair
    out = torch.stack([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)   # (B, h, T, d/2, 2)
    return out.flatten(-2)                             # (B, h, T, d)


cos, sin = rope_freqs(head_dim=8, max_len=64)
q = torch.randn(1, 1, 6, 8)
q_rot = apply_rope(q, cos, sin)
print("rope preserves shape:", tuple(q.shape), "->", tuple(q_rot.shape))
print("and preserves norms (rotations are orthogonal):",
      torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5))

# %% [markdown]
# ### Verify the defining property
#
# The claim is that the rotated dot product depends only on the offset. Take the
# same two vectors at positions `(2, 5)` and at `(20, 23)` — same distance 3 —
# and the scores should match exactly.

# %%
qv = torch.randn(1, 1, 1, 8)
kv = torch.randn(1, 1, 1, 8)


def score_at(pos_q: int, pos_k: int) -> float:
    cq = cos[pos_q:pos_q + 1], sin[pos_q:pos_q + 1]
    ck = cos[pos_k:pos_k + 1], sin[pos_k:pos_k + 1]
    qr = apply_rope(qv, *cq)
    kr = apply_rope(kv, *ck)
    return (qr @ kr.transpose(-2, -1)).item()


print(f"positions (2, 5),   distance 3 -> {score_at(2, 5):+.6f}")
print(f"positions (20, 23), distance 3 -> {score_at(20, 23):+.6f}")
print(f"positions (7, 8),   distance 1 -> {score_at(7, 8):+.6f}")
print("\nSame distance => same score, anywhere in the sequence. That is RoPE.")

# %% [markdown]
# ### RoPE attention, and the `base` knob
#
# `base=10000` sets how fast the slowest dial turns. Raising it ("NTK scaling")
# is the standard trick for stretching a trained model to longer contexts,
# because it slows every dial down so positions beyond training still fall in a
# seen range.

# %%
class RoPEAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, base=10000.0, dropout=0.0):
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        # generous max_len: RoPE has no learned table, so this is free
        cos, sin = rope_freqs(self.d, block_size * 8, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, Cx = x.shape
        q, k, v = self.qkv(x).split(Cx, dim=2)
        q = q.view(B, T, self.h, self.d).transpose(1, 2)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)
        q = apply_rope(q, self.cos, self.sin)          # position enters HERE,
        k = apply_rope(k, self.cos, self.sin)          # not in the embeddings
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, Cx)
        return self.drop(self.proj(y))


rope_attn = RoPEAttention(64, 4, 32)
print("out:", tuple(rope_attn(torch.randn(2, 20, 64)).shape))
print("no positional PARAMETERS:",
      [n for n, _ in rope_attn.named_parameters()])
print("\nAlso note: T=200 works even though block_size=32, because there is no")
print("table to overflow —", tuple(rope_attn(torch.randn(1, 200, 64)).shape))

# %% [markdown]
# ## 5 · ALiBi — attention with linear biases
#
# The simplest scheme of all: add a penalty to the attention scores that grows
# linearly with distance, with a different slope per head.
#
# ```
# scores[i, j] += -slope_h * (i - j)
# ```
#
# No parameters, no embeddings. Each head gets a slope from a geometric series,
# so some heads look locally and others globally. It extrapolates remarkably
# well, because "further away = more penalty" is true at any length.

# %%
def alibi_slopes(n_heads: int) -> torch.Tensor:
    """Geometric series 1/2^1, 1/2^2, ... (the paper's construction)."""
    start = 2 ** (-8 / n_heads)
    return torch.tensor([start ** (i + 1) for i in range(n_heads)])


def alibi_bias(n_heads: int, T: int) -> torch.Tensor:
    """(1, h, T, T) additive bias; 0 on the diagonal, more negative further back."""
    slopes = alibi_slopes(n_heads).view(n_heads, 1, 1)
    dist = torch.arange(T).view(1, -1) - torch.arange(T).view(-1, 1)   # j - i
    return (slopes * dist.clamp(max=0).float()).unsqueeze(0)


print("slopes for 8 heads:", alibi_slopes(8).tolist())
bias = alibi_bias(4, 6)
print("\nhead 0 bias matrix (upper triangle is masked away later):")
print(bias[0, 0])

# %%
class ALiBiAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("slopes", alibi_slopes(n_heads).view(1, n_heads, 1, 1),
                             persistent=False)

    def forward(self, x):
        B, T, Cx = x.shape
        q, k, v = self.qkv(x).split(Cx, dim=2)
        q = q.view(B, T, self.h, self.d).transpose(1, 2)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.d ** -0.5            # (B,h,T,T)
        dist = (torch.arange(T, device=x.device).view(1, -1)
                - torch.arange(T, device=x.device).view(-1, 1))
        att = att + self.slopes * dist.clamp(max=0).float()          # ALiBi
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        att = att.masked_fill(~causal, float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, Cx)
        return self.drop(self.proj(y))


alibi = ALiBiAttention(64, 4)
print("out:", tuple(alibi(torch.randn(2, 20, 64)).shape),
      "| any length:", tuple(alibi(torch.randn(1, 500, 64)).shape))

# %% [markdown]
# ### What the bias does to a uniform attention pattern
#
# With all-equal scores, ALiBi turns attention into an exponential decay over
# distance whose sharpness is set by the head's slope.

# %%
T = 32
for hd, slope in enumerate(alibi_slopes(4)):
    scores = torch.zeros(T)
    dist = torch.arange(T) - (T - 1)
    w = F.softmax(scores + slope * dist.float(), dim=-1)
    eff = 1.0 / (w ** 2).sum()          # effective number of attended positions
    print(f"  head {hd}: slope {slope:.4f} -> attends to ~{eff:.1f} of {T} positions")

# %% [markdown]
# ## 6 · The experiment that matters: length extrapolation
#
# Train three identical models — learned absolute, RoPE, ALiBi — at
# `block_size=64`, then evaluate at 64, 128 and 256. Only relative schemes can
# even *run* past their training length.

# %%
from tinygpt import GPTConfig, Block as GPTBlock


class PosLM(nn.Module):
    """Same skeleton, swappable positional scheme."""

    def __init__(self, vocab_size, scheme, d_model=128, n_heads=4, n_layers=4,
                 block_size=64, dropout=0.1):
        super().__init__()
        self.scheme, self.block_size = scheme, block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if scheme == "learned":
            self.pos_emb = nn.Embedding(block_size, d_model)
        elif scheme == "sinusoidal":
            self.register_buffer("pe", sinusoidal_encoding(block_size * 8, d_model),
                                 persistent=False)

        def make_attn():
            if scheme == "rope":
                return RoPEAttention(d_model, n_heads, block_size, dropout=dropout)
            if scheme == "alibi":
                return ALiBiAttention(d_model, n_heads, dropout=dropout)
            cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size * 8,
                            d_model=d_model, n_heads=n_heads, dropout=dropout)
            return GPTBlock(cfg).attn

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": make_attn(),
                "ln2": nn.LayerNorm(d_model),
                "ff": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                    nn.Linear(4 * d_model, d_model),
                                    nn.Dropout(dropout)),
            }))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        # GPT-2 style init (notebook 05) so the initial loss lands near ln(V)
        self.apply(lambda m: (
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, (nn.Linear, nn.Embedding)) else None))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.scheme == "learned":
            # There is no table row past block_size, so the only way to run at
            # all is to CLAMP: every position beyond the training length reuses
            # the last row. The model runs and is confidently wrong about order.
            pos = torch.arange(T, device=idx.device).clamp(max=self.block_size - 1)
            x = x + self.pos_emb(pos)
        elif self.scheme == "sinusoidal":
            # The original paper multiplies embeddings by sqrt(d_model) before
            # adding the encoding, and this is why: the fixed encoding has
            # entries in [-1, 1], while GPT-2-style embeddings are initialized
            # with std 0.02. Without the rescale the position signal drowns the
            # token identity and training is dramatically worse. Try removing it.
            x = x * math.sqrt(x.size(-1)) + self.pe[:T]
        for layer in self.layers:
            x = x + layer["attn"](layer["ln1"](x))
            x = x + layer["ff"](layer["ln2"](x))
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss


text = cu.get_text(verbose=False)
tok = cu.CharTokenizer(text)
train_data, val_data = cu.train_val_split(tok.encode(text))
splits = {"train": train_data, "val": val_data}
TRAIN_T = 64

results = {}
for scheme in ("learned", "sinusoidal", "rope", "alibi"):
    cu.set_seed(1337)
    m = PosLM(tok.vocab_size, scheme, block_size=TRAIN_T).to(device)
    print(f"\n=== {scheme} ({cu.human(cu.count_params(m))} params) ===")
    hist = cu.train(m, splits, steps=500, batch_size=32, block_size=TRAIN_T,
                    lr=1e-3, device=device, eval_every=250, eval_iters=15)
    results[scheme] = {"model": m, "train_val": hist["val"][-1]}

# %%
print(f"{'scheme':<12}" + "".join(f"{'T=' + str(t):>12}" for t in (64, 128, 256)))
for scheme, r in results.items():
    row = f"{scheme:<12}"
    for T_eval in (64, 128, 256):
        try:
            ev = cu.estimate_loss(r["model"], {"val": val_data}, 8, T_eval,
                                  device, iters=10)
            row += f"{ev['val']:>12.4f}"
        except Exception as exc:
            row += f"{type(exc).__name__:>12}"
    print(row)

print(f"\nAll models trained at T={TRAIN_T}. Reading the table:")
print("  learned     : only runs because we CLAMP out-of-range positions to the")
print("                last table row; every token past T=64 claims to be at")
print("                position 63, so the loss falls apart. Without the clamp")
print("                it would raise an IndexError — a hard architectural wall.")
print("  sinusoidal  : runs at any length, but the network never saw those")
print("                encoding values, so quality drops.")
print("  rope/alibi  : degrade most gracefully. ALiBi usually wins this test;")
print("                RoPE wins in practice once you add NTK/YaRN scaling.")

# %% [markdown]
# ## 7 · Choosing, in 2026
#
# - **New decoder-only LM?** RoPE. It is what every open model uses, it costs no
#   parameters, and the context-extension literature (NTK, YaRN, position
#   interpolation) is all built on it.
# - **Need extreme length robustness with no tricks?** ALiBi.
# - **Fixed-length encoder (classification, ViT)?** Learned absolute is fine and
#   marginally simpler.
# - **Sinusoidal?** Historical interest, and a nice thing to have derived once.

# %% [markdown]
# ## Exercises
#
# 1. **NTK scaling.** Take the trained RoPE model and re-evaluate at `T=256`
#    with `base` raised to 40000 (rebuild the cos/sin buffers, don't retrain).
#    Does the loss improve?
# 2. **Position interpolation.** Instead of raising `base`, divide the position
#    indices by 4 before computing angles. Compare against NTK scaling.
# 3. **Learn the slopes.** Make ALiBi's slopes `nn.Parameter`s. Do they stay in
#    a geometric series after training? What do the learned values say about how
#    much local context this corpus needs?
# 4. **Probe the positions.** Train a linear classifier to predict a token's
#    absolute position from its residual-stream vector, for each scheme. Which
#    representations actually carry absolute position?
# 5. **RoPE by hand.** Implement RoPE with complex numbers
#    (`torch.view_as_complex`) and assert it matches `apply_rope` to 1e-6. This
#    is how it's often written in production code.
#
# ---
# **Next:** [08 · Modern architecture upgrades: RMSNorm, SwiGLU, GQA, flash attention](08_modern_architecture.ipynb)
