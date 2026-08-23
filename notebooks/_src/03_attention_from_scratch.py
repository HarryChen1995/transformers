# %% [markdown]
# # 03 · Attention from scratch
#
# **Goal:** derive self-attention instead of being handed the formula. We start
# with "average the past tokens", make the average weighted, then make the
# weights *data-dependent* — which is attention.
#
# **You will be able to:**
# 1. explain why the causal average can be written as a matrix multiply
# 2. say what each of query, key and value is for, in one sentence each
# 3. write scaled dot-product attention from memory, with correct shapes
# 4. justify the `1/sqrt(d)` scaling and the `-inf` mask numerically
# 5. read an attention heatmap
#
# Prerequisites: notebooks 01–02.

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
torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

# %% [markdown]
# ## 1 · The problem, restated
#
# The bigram model predicts from one token. We want each position `t` to build a
# representation from **all** positions `<= t`. Two hard constraints:
#
# - **Causality.** Position `t` may never see `t+1`; otherwise the model cheats
#   at training time and is useless at generation time.
# - **Variable, learned relevance.** Which earlier tokens matter depends on the
#   content, not just the distance. A verb needs its subject, wherever it sits.
#
# Start with the crudest version that satisfies causality: average everything
# up to and including `t`.

# %%
B, T, C = 1, 5, 2                       # 1 sequence, 5 positions, 2 channels
x = torch.arange(1.0, T * C + 1).view(B, T, C)
print("x =\n", x[0])

# Version 1: explicit loops. Correct, and hopelessly slow.
out_loop = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        out_loop[b, t] = x[b, :t + 1].mean(dim=0)     # mean over the past
print("\ncausal running mean =\n", out_loop[0])

# %% [markdown]
# ## 2 · The matrix trick
#
# A weighted sum over positions **is** a matrix multiply. Build a lower
# triangular matrix `W` of shape `(T, T)` whose row `t` is `1/(t+1)` on the first
# `t+1` entries and 0 after. Then `W @ x` computes all the running means at once.
#
# The zeros above the diagonal are what enforce causality: row `t` has no weight
# on any column `> t`.

# %%
W = torch.tril(torch.ones(T, T))
W = W / W.sum(dim=1, keepdim=True)      # normalize each row to sum to 1
print("W =\n", W)

out_mat = W @ x                          # (T, T) @ (B, T, C) -> (B, T, C)
print("\nW @ x =\n", out_mat[0])
print("\nmatches the loop:", torch.allclose(out_loop, out_mat, atol=1e-6))

# %% [markdown]
# ## 3 · From averaging to softmax weights
#
# Uniform weights throw away information. Let's instead produce arbitrary
# **scores** and turn them into weights with a softmax — masking future
# positions with `-inf` so that `exp(-inf) = 0` makes them exactly zero after
# normalization.
#
# Using `-inf` before softmax (rather than multiplying by zero afterwards) is
# what keeps the surviving weights correctly renormalized.

# %%
scores = torch.zeros(T, T)               # all-zero scores == uniform attention
mask = torch.tril(torch.ones(T, T))
masked = scores.masked_fill(mask == 0, float("-inf"))
print("masked scores =\n", masked)

W2 = F.softmax(masked, dim=-1)
print("\nsoftmax(masked) =\n", W2)
print("\nidentical to the uniform average:", torch.allclose(W, W2))

# %% [markdown]
# So the causal average is just the special case of attention where all scores
# are equal. Everything left to do is: **make the scores depend on the data.**

# %% [markdown]
# ## 4 · Query, key, value
#
# Each token emits three vectors, all linear projections of its own embedding:
#
# | | question it answers | used as |
# |---|---|---|
# | **query** `q_i` | "what am I looking for?" | the row of the score matrix |
# | **key** `k_j` | "what do I offer?" | the column of the score matrix |
# | **value** `v_j` | "what do I contribute if selected?" | what gets summed |
#
# The affinity of token `i` for token `j` is the dot product `q_i · k_j`: large
# when the query and key point the same way. Crucially the *content* decides,
# and the projections are learned.
#
# ```
# scores = Q @ K.T / sqrt(d)     (T, T)   who attends to whom
# scores = mask(scores)          (T, T)   no peeking at the future
# weights = softmax(scores)      (T, T)   rows sum to 1
# out = weights @ V              (T, d)   weighted sum of values
# ```

# %%
cu.set_seed(0)
B, T, C = 1, 6, 16
head_dim = 8
x = torch.randn(B, T, C)

W_q = nn.Linear(C, head_dim, bias=False)
W_k = nn.Linear(C, head_dim, bias=False)
W_v = nn.Linear(C, head_dim, bias=False)

q = W_q(x)                              # (B, T, d)
k = W_k(x)                              # (B, T, d)
v = W_v(x)                              # (B, T, d)
print("q, k, v:", tuple(q.shape), tuple(k.shape), tuple(v.shape))

scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)      # (B, T, T)
print("scores:", tuple(scores.shape))
print(scores[0])

# %% [markdown]
# ### Why divide by `sqrt(d)`?
#
# For random unit-variance `q` and `k`, the dot product over `d` dimensions has
# variance `d`. As `d` grows, scores spread out, softmax saturates into a
# one-hot, and the gradient through it vanishes. Dividing by `sqrt(d)` restores
# unit variance — for any head size.

# %%
for d in (4, 64, 1024):
    qq, kk = torch.randn(4096, d), torch.randn(4096, d)
    raw = (qq * kk).sum(-1)
    print(f"d={d:>5}: std(q·k) = {raw.std():7.2f} | "
          f"scaled = {(raw / math.sqrt(d)).std():.2f}")

print("\neffect on softmax sharpness (max weight of one row):")
row = torch.randn(64)
for scale, name in ((1.0, "unscaled, d=64"), (1 / math.sqrt(64), "scaled")):
    w = F.softmax(row * 8 * scale, dim=-1)
    print(f"  {name:>16}: max weight {w.max():.3f}, entropy {-(w * w.log()).sum():.3f}")

# %% [markdown]
# ### Mask, normalize, aggregate

# %%
tril = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(tril == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)                  # (B, T, T)
print("attention weights (row t sums to 1, zeros to the right of the diagonal):")
print(weights[0])
print("\nrow sums:", weights[0].sum(-1))

out = weights @ v                                    # (B, T, T) @ (B, T, d)
print("\noutput:", tuple(out.shape))

# %% [markdown]
# Note that position 0 attends only to itself (its row is `[1, 0, 0, ...]`) — it
# has no past. This is also why the very first token's output is just a linear
# map of itself.

# %% [markdown]
# ## 5 · One attention head as a Module
#
# Same math, packaged. This is the version you will recognize in real codebases.

# %%
class SelfAttentionHead(nn.Module):
    """Single-head causal self-attention: (B, T, C) -> (B, T, head_dim)."""

    def __init__(self, d_model: int, head_dim: int, block_size: int,
                 dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.key = nn.Linear(d_model, head_dim, bias=False)
        self.query = nn.Linear(d_model, head_dim, bias=False)
        self.value = nn.Linear(d_model, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.head_dim = head_dim
        # a buffer, not a parameter: never trained, but moves with .to(device)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, return_weights: bool = False):
        B, T, C = x.shape
        q = self.query(x)                                  # (B, T, d)
        k = self.key(x)                                    # (B, T, d)
        v = self.value(x)                                  # (B, T, d)

        att = q @ k.transpose(-2, -1) * self.head_dim ** -0.5      # (B, T, T)
        if self.causal:
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)                       # (B, T, T)
        out = self.dropout(att) @ v                        # (B, T, d)
        return (out, att) if return_weights else out


head = SelfAttentionHead(d_model=16, head_dim=8, block_size=32)
out, att = head(torch.randn(2, 6, 16), return_weights=True)
print("out:", tuple(out.shape), "| att:", tuple(att.shape))
print("params:", cu.count_params(head), "| buffers:", [n for n, _ in head.named_buffers()])

# %% [markdown]
# ### PyTorch's fused version
#
# `F.scaled_dot_product_attention` does all four steps in one fused kernel
# (FlashAttention when available), never materializing the `(T, T)` matrix.
# It's faster and far more memory-efficient — but you cannot inspect the
# weights, which is why we keep the manual version for teaching. Notebook 08
# benchmarks the difference.

# %%
q_, k_, v_ = (torch.randn(2, 4, 6, 8) for _ in range(3))          # (B, h, T, d)
fused = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)

manual = q_ @ k_.transpose(-2, -1) / math.sqrt(8)
manual = manual.masked_fill(torch.tril(torch.ones(6, 6)) == 0, float("-inf"))
manual = F.softmax(manual, dim=-1) @ v_
print("fused == manual:", torch.allclose(fused, manual, atol=1e-5))

# %% [markdown]
# ## 6 · Look at what a *trained* head does
#
# Random weights give near-uniform attention, which tells us nothing. Let's
# train a one-head language model for a couple of minutes and inspect it. Note
# the model still has no positional information yet — that's notebook 07 — so it
# can only learn content-based relevance.

# %%
text = cu.get_text(verbose=False)
tok = cu.CharTokenizer(text)
train_data, val_data = cu.train_val_split(tok.encode(text))
splits = {"train": train_data, "val": val_data}
device = cu.get_device()
block_size = 32


class OneHeadLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, block_size=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.head = SelfAttentionHead(d_model, d_model, block_size)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, return_weights=False):
        x = self.tok_emb(idx)                                # (B, T, C)
        h = self.head(x, return_weights=return_weights)
        h, att = h if return_weights else (h, None)
        logits = self.out(self.ln(x + h))                    # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return (logits, loss, att) if return_weights else (logits, loss)


cu.set_seed(1337)
model = OneHeadLM(tok.vocab_size, 64, block_size)
print("params:", cu.human(cu.count_params(model)))
hist = cu.train(model, splits, steps=800, batch_size=32, block_size=block_size,
                lr=3e-3, device=device, eval_every=200)

# %% [markdown]
# One attention head already beats the bigram model (~1.90 val loss in notebook
# 02), while using context it *chose* to use.

# %%
prompt = "the king speaks of"
idx = torch.tensor([tok.encode(prompt)], device=device)
model.eval()
with torch.no_grad():
    _, _, att = model(idx, return_weights=True)

try:
    import matplotlib.pyplot as plt
    cu.show_attention(att[0], tokens=list(prompt), title="trained single head")
    plt.tight_layout(); plt.show()
except ImportError:
    print(att[0])

# %% [markdown]
# ### How to read that plot
#
# Row `t` = the query at position `t`; bright cells = keys it pulled from.
# Everything above the diagonal is black — the causal mask. Look for columns
# that are bright across many rows: those are tokens the whole sequence cares
# about (spaces and newlines usually, at char level).

# %%
with torch.no_grad():
    _, _, att = model(idx, return_weights=True)
a = att[0].cpu()
print("attention entropy per query position (low = focused, high = diffuse):")
for t in range(a.size(0)):
    row = a[t, :t + 1]
    ent = -(row * row.clamp_min(1e-9).log()).sum().item()
    focus = tok.decode([idx[0, int(row.argmax())].item()])
    print(f"  t={t:>2} {prompt[t]!r:>5}  entropy {ent:5.2f}  peak on {focus!r}")

# %% [markdown]
# ## 7 · Self-attention vs cross-attention, causal vs bidirectional
#
# The same four lines cover every variant; only *where Q, K, V come from* and
# *what is masked* change:
#
# | variant | Q from | K, V from | mask | used in |
# |---|---|---|---|---|
# | causal self-attention | x | x | triangular | GPT decoders (nb 04–05) |
# | bidirectional self-attention | x | x | padding only | BERT encoders (nb 11) |
# | cross-attention | decoder | encoder | padding only | translation (nb 10) |
#
# Bidirectional attention lets every token see every other token — great for
# understanding, unusable for generation.

# %%
x_enc = torch.randn(1, 5, 16)
bi = SelfAttentionHead(16, 8, 32, causal=False)
_, w_bi = bi(x_enc, return_weights=True)
print("bidirectional row 0 (attends everywhere):", w_bi[0, 0])
_, w_causal = head(x_enc, return_weights=True)
print("causal row 0        (attends to itself):", w_causal[0, 0])

# %% [markdown]
# ## 8 · Cost
#
# Attention compares every position with every position: `O(T² · d)` time and,
# in the naive implementation, an `O(T²)` matrix per head per layer. Doubling
# context quadruples the work. This single fact drives most modern
# architecture research (flash attention, sliding windows, GQA — notebook 08).

# %%
d_model, n_heads, n_layers = 512, 8, 12
for T_ in (128, 512, 2048, 8192):
    scores_per_layer = n_heads * T_ * T_
    mb = scores_per_layer * 4 / 1e6                       # fp32 bytes
    print(f"T={T_:>5}: {scores_per_layer:>12,} score entries/layer "
          f"= {mb:8.1f} MB, {mb * n_layers / 1000:6.2f} GB for {n_layers} layers")

# %% [markdown]
# ## Exercises
#
# 1. **Break causality.** Set `causal=False` in `OneHeadLM` and retrain. The loss
#    will collapse toward zero. Explain exactly which tensor leaks the answer.
# 2. **Remove the scaling.** Delete `* self.head_dim ** -0.5` and retrain with
#    `head_dim=64`. Plot the max attention weight per row before and after.
# 3. **Weights, not values.** Replace `weights @ v` with `weights @ x` (attend
#    over raw embeddings, no value projection). How much worse, and why is the
#    value projection worth its parameters?
# 4. **Distance profile.** For the trained head, compute average attention weight
#    as a function of `t - j` and plot it. Does the head prefer nearby tokens
#    even though it has no positional information? Why might that happen?
# 5. **Two heads by hand.** Instantiate two `SelfAttentionHead(64, 32, ...)`,
#    concatenate their outputs, and confirm you get `(B, T, 64)`. You have just
#    written multi-head attention — notebook 04 makes it efficient.
#
# ---
# **Next:** [04 · Multi-head attention and the transformer block](04_multihead_attention_and_blocks.ipynb)
