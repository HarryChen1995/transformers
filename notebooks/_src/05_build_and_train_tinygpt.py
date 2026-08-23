# %% [markdown]
# # 05 · Build and train TinyGPT end to end
#
# **Goal:** assemble a complete, correct GPT — the milestone of the first half of
# the course. This is the notebook that corresponds to `transformer.py` in the
# repo root, but done properly: positional embeddings, sane initialization,
# weight tying, train/val monitoring, checkpointing and generation.
#
# **You will be able to:**
# 1. explain why a transformer needs positional information at all
# 2. initialize a GPT the way GPT-2 does, and say why the scaled init exists
# 3. tie the embedding and output head, and know what it saves
# 4. train with a validation loop and read overfitting off the curves
# 5. generate text with temperature and top-k, and save/load a checkpoint
#
# Prerequisites: notebooks 01–04.

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

cu.set_seed(1337)
device = cu.get_device()
print("device:", cu.describe_device(device))

# %% [markdown]
# ## 1 · Attention is permutation-equivariant — that's a problem
#
# Attention is a weighted *sum* over positions. Sums don't care about order.
# Shuffle the input tokens and every output is shuffled the same way, with
# identical values. So the model literally cannot distinguish "dog bites man"
# from "man bites dog".
#
# Let's prove it on the block from notebook 04.

# %%
from torch import nn as _nn


class MiniAttn(_nn.Module):
    def __init__(self, C=8):
        super().__init__()
        self.qkv = _nn.Linear(C, 3 * C, bias=False)

    def forward(self, x):
        C = x.size(-1)
        q, k, v = self.qkv(x).split(C, dim=-1)
        att = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(C), dim=-1)
        return att @ v


torch.manual_seed(0)
attn = MiniAttn()
x = torch.randn(1, 4, 8)
perm = torch.tensor([2, 0, 3, 1])

out_then_perm = attn(x)[:, perm]
perm_then_out = attn(x[:, perm])
print("attention(permuted x) == permute(attention(x)):",
      torch.allclose(out_then_perm, perm_then_out, atol=1e-6))
print("\n=> order information must be INJECTED. That is what positional")
print("   embeddings do. (The causal mask constrains *which* positions are")
print("   visible, but says nothing about how far apart they are.)")

# %% [markdown]
# ## 2 · Learned absolute positional embeddings
#
# The GPT-2 solution is the simplest one that works: a second embedding table
# indexed by *position*, added to the token embedding.
#
# ```
# x = tok_emb(idx) + pos_emb(arange(T))     # (B,T,C) + (T,C) -> (B,T,C)
# ```
#
# Two consequences worth knowing now: the model can only ever handle
# `T <= block_size` (there is no row for position 1025), and positions are
# absolute, so nothing generalizes to longer contexts. Notebook 07 fixes both
# with sinusoidal encodings, RoPE and ALiBi.

# %%
block_size, d_model = 8, 6
pos_emb = nn.Embedding(block_size, d_model)
pos = torch.arange(block_size)
print("pos ids:", pos.tolist())
print("pos_emb(pos):", tuple(pos_emb(pos).shape), "-> broadcasts over the batch")

# %% [markdown]
# ## 3 · The full model
#
# Every piece below was built in notebooks 03–04. What is new here is the
# scaffolding around them: embeddings, final norm, the LM head, initialization,
# weight tying, and `generate`.

# %%
@dataclass
class GPTConfig:
    vocab_size: int = 47
    block_size: int = 128
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True

    @property
    def head_dim(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.dropout_p = cfg.dropout
        self.register_buffer("tril", torch.tril(
            torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, -1))

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, h, d).transpose(1, 2)                    # (B,h,T,d)
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * d ** -0.5               # (B,h,T,T)
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = self.attn_drop(att) @ v                               # (B,h,T,d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)          # (B,T,C)
        y = self.resid_drop(self.proj(y))
        return (y, att) if return_weights else y


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            # one shared (V, C) matrix used both to embed and to unembed
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 trick: scale down the projections that write into the residual
        # stream by 1/sqrt(2L), so the stream's variance doesn't grow with depth
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_weights=False):
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)                   # (T,)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))       # (B,T,C)

        attentions = []
        for block in self.blocks:
            if return_weights:
                y, att = block.attn(block.ln1(x), return_weights=True)
                x = x + y
                x = x + block.ff(block.ln2(x))
                attentions.append(att)
            else:
                x = block(x)
        x = self.ln_f(x)                                           # (B,T,C)
        logits = self.head(x)                                      # (B,T,V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return (logits, loss, attentions) if return_weights else (logits, loss)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            # crop to the last block_size tokens: the model has no positional
            # embedding rows beyond that
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = cu.filter_logits(logits[:, -1, :], temperature, top_k, top_p)
            probs = F.softmax(logits, dim=-1)                      # (B,V)
            nxt = torch.multinomial(probs, num_samples=1)           # (B,1)
            idx = torch.cat([idx, nxt], dim=1)
        self.train(was_training)
        return idx


# %% [markdown]
# ### Weight tying
#
# `tok_emb` maps a token id to a vector; `head` maps a vector back to scores over
# tokens. Both are `(V, C)` matrices describing "what each token looks like", so
# sharing one matrix is principled, not just thrifty — and it removes `V·C`
# parameters, which at GPT-2's `V=50257` is 39M of 124M.

# %%
text = cu.get_text(verbose=False)
tok = cu.CharTokenizer(text)
train_data, val_data = cu.train_val_split(tok.encode(text))
splits = {"train": train_data, "val": val_data}
print(tok, "| train", tuple(train_data.shape), "| val", tuple(val_data.shape))

cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=128, d_model=192,
                n_heads=6, n_layers=6, dropout=0.1)

tied = TinyGPT(cfg)
untied = TinyGPT(GPTConfig(**{**asdict(cfg), "tie_weights": False}))
print("\ntied  :", cu.human(cu.count_params(tied)), "params")
print("untied:", cu.human(cu.count_params(untied)), "params")
print("shared storage:", tied.head.weight.data_ptr() == tied.tok_emb.weight.data_ptr())

# %% [markdown]
# ### Check the initialization before you train
#
# Two cheap checks that catch most init bugs:
# 1. initial loss should be near `ln(V)`
# 2. activations should not explode as they pass through the block stack

# %%
model = TinyGPT(cfg)
xb, yb = cu.get_batch(train_data, 16, cfg.block_size)
with torch.no_grad():
    _, loss0 = model(xb, yb)
print(f"initial loss {loss0.item():.4f} vs ln(V) = {math.log(tok.vocab_size):.4f}")

with torch.no_grad():
    z = model.drop(model.tok_emb(xb) + model.pos_emb(torch.arange(cfg.block_size)))
    print(f"{'after embeddings':>22}: std {z.std():.3f}")
    for i, blk in enumerate(model.blocks):
        z = blk(z)
        print(f"{'after block ' + str(i):>22}: std {z.std():.3f}")
print("\nGrowth of ~sqrt(depth) is expected and healthy; a 10x jump per block is not.")

# %% [markdown]
# ## 4 · Train
#
# The loop is the same five lines from notebook 01, plus periodic evaluation on
# both splits. Notebook 09 adds LR schedules, mixed precision, gradient
# accumulation and `torch.compile`.
#
# On the CUDA desktop, raise `steps` to a few thousand and `d_model`/`n_layers`
# — the same code runs unchanged. See `scripts/train_gpt.py` for the script form.

# %%
cu.set_seed(1337)
model = TinyGPT(cfg).to(device)
print("params:", cu.human(cu.count_params(model)))

steps = 1500 if device.type != "cpu" else 500
t0 = time.time()
history = cu.train(model, splits, steps=steps, batch_size=32,
                   block_size=cfg.block_size, lr=3e-4, device=device,
                   eval_every=max(steps // 6, 1), eval_iters=20)
print(f"\ntrained {steps} steps in {time.time() - t0:.1f}s")

# %%
try:
    import matplotlib.pyplot as plt
    cu.plot_history(history, f"TinyGPT ({cu.human(cu.count_params(model))} params)")
    plt.tight_layout(); plt.show()
except ImportError:
    pass

gap = history["val"][-1] - history["train"][-1]
print(f"train {history['train'][-1]:.4f} | val {history['val'][-1]:.4f} | gap {gap:+.4f}")
print("\nReading the curves:")
print("  both still falling      -> undertrained, keep going")
print("  val flat, train falling -> overfitting: more data, more dropout, or stop")
print("  both flat and high      -> underfitting: bigger model or higher LR")

# %% [markdown]
# ## 5 · Generate
#
# `generate` feeds the model its own output one token at a time, cropping the
# context to `block_size`. Two knobs, both from notebook 02's temperature idea:
#
# - **temperature** — flatten (`>1`) or sharpen (`<1`) the distribution
# - **top-k** — keep only the `k` most likely tokens, renormalize
#
# It is also the *slow* way to sample: each new token re-runs the whole context.
# Notebook 12 fixes that with a KV cache.

# %%
prompt = "The king"
idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

for temperature, top_k in ((0.5, None), (0.8, 20), (1.0, None), (1.4, None)):
    out = model.generate(idx, 200, temperature=temperature, top_k=top_k)
    print(f"--- temperature={temperature}, top_k={top_k} ---")
    print(tok.decode(out[0]))
    print()

# %% [markdown]
# ### Single next-token prediction
#
# Sometimes you want the distribution, not a sample — for evaluation, for
# debugging, or to see how confident the model is.

# %%
with torch.no_grad():
    logits, _ = model(idx)
probs = F.softmax(logits[0, -1], dim=-1)
top = torch.topk(probs, 8)
print(f"after {prompt!r}, the model's top continuations:")
for p, i in zip(top.values, top.indices):
    print(f"  {tok.itos[int(i)]!r:>6}  {p.item():.3f}  {'#' * int(p.item() * 50)}")
print(f"\nentropy {-(probs * probs.clamp_min(1e-9).log()).sum():.3f} nats "
      f"(max possible {math.log(tok.vocab_size):.3f})")

# %% [markdown]
# ## 6 · What the trained heads look like
#
# With positional embeddings in place, heads specialize by *position* as well as
# content. Typical char-level patterns: a "previous token" head (bright band just
# below the diagonal), a "start of line" head (bright column at newlines), and
# diffuse heads that average broadly.

# %%
probe = "The king speaks of the crown."
idx_probe = torch.tensor([tok.encode(probe)], device=device)
model.eval()
with torch.no_grad():
    _, _, attentions = model(idx_probe, return_weights=True)
print("attention tensors per layer:", tuple(attentions[0].shape), "= (B, h, T, T)")

try:
    import matplotlib.pyplot as plt
    layer = len(attentions) - 1
    A = attentions[layer][0].cpu()
    n_show = min(4, A.size(0))
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    for hd in range(n_show):
        cu.show_attention(A[hd], tokens=list(probe), ax=axes[hd],
                          title=f"layer {layer}, head {hd}")
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %% [markdown]
# ### Quantify what each head does
#
# A cheap, surprisingly informative statistic: the average *distance* a head
# attends back, and how much mass it puts on the immediately preceding token.

# %%
T = idx_probe.size(1)
dist = (torch.arange(T).view(-1, 1) - torch.arange(T).view(1, -1)).float()
print(f"{'layer':>5} {'head':>5} {'mean dist':>10} {'prev-token mass':>16}")
for li, A in enumerate(attentions):
    for hd in range(A.size(1)):
        w = A[0, hd].cpu()
        mean_dist = (w * dist).sum() / w.sum()
        prev = torch.diagonal(w, offset=-1).mean()
        print(f"{li:>5} {hd:>5} {mean_dist:>10.2f} {prev:>16.3f}")

# %% [markdown]
# ## 7 · Save and load
#
# Save the `state_dict` *and* the config — weights without the config that
# produced their shapes are useless.

# %%
path = cu.save_checkpoint(model, "tinygpt", config=asdict(cfg),
                          vocab=tok.chars, history=history)

ckpt = cu.load_checkpoint("tinygpt")
restored = TinyGPT(GPTConfig(**ckpt["config"])).to(device)
restored.load_state_dict(ckpt["state_dict"])
restored.eval()
with torch.no_grad():
    _, l_new = restored(*cu.get_batch(val_data, 16, cfg.block_size, device))
    _, l_old = model(*cu.get_batch(val_data, 16, cfg.block_size, device))
print(f"reloaded model works (val batch loss {l_new.item():.4f} vs {l_old.item():.4f})")

# %% [markdown]
# ## 8 · Relationship to `transformer.py`
#
# The repo root has `transformer.py`: the same architecture as a single runnable
# script, trained on a three-line corpus. The differences here are all things a
# real training run needs:
#
# | | `transformer.py` | this notebook |
# |---|---|---|
# | corpus | 3 hardcoded lines | full text file, 90/10 split |
# | validation | none | monitored every N steps |
# | init | PyTorch defaults | GPT-2 style + residual scaling |
# | weight tying | no | yes |
# | sampling | temperature + top-k | + nucleus, via `cu.filter_logits` |
# | checkpointing | none | config + weights + history |
#
# Read `transformer.py` now — it should feel completely transparent.

# %%
print((ROOT / "transformer.py").read_text()[:600], "...")

# %% [markdown]
# ## 9 · Scaling this up on a real GPU
#
# `scripts/train_gpt.py` is this notebook's model in script form, with CLI flags.
# On a CUDA box:
#
# ```bash
# ssh gpu-desktop
# cd transformer
# python scripts/train_gpt.py --steps 5000 --d_model 384 --n_layers 6 \
#     --block_size 256 --batch_size 64 --amp --compile
# ```
#
# Then copy `checkpoints/*.pt` back and load it here with `cu.load_checkpoint`.
# Notebook 09 covers what each of those flags actually does.

# %% [markdown]
# ## Exercises
#
# 1. **No positions.** Set `pos_emb` output to zero and retrain. How much loss
#    does the model give up? (At char level, less than you'd guess — think about
#    why.)
# 2. **Depth vs width.** Compare `(d_model=384, n_layers=3)` against
#    `(d_model=192, n_layers=12)` at matched parameter count. Which wins here,
#    and does the answer depend on the amount of data?
# 3. **Overfit on purpose.** Train on only the first 20,000 characters with
#    `dropout=0`. Watch val loss turn upward while train loss keeps falling —
#    then add dropout back and confirm the gap shrinks.
# 4. **Kill the scaled init.** Remove the `1/sqrt(2L)` projection re-init, set
#    `n_layers=16`, and print activation std per block as in section 3.
# 5. **Context extrapolation.** Generate with a prompt longer than `block_size`.
#    Explain the crop in `generate`, then try to make the model handle longer
#    contexts without retraining. (You can't — that's notebook 07's ALiBi/RoPE.)
#
# ---
# **Next:** [06 · Tokenization and byte-pair encoding](06_tokenization_and_bpe.ipynb) —
# stop wasting sequence length on individual characters.
