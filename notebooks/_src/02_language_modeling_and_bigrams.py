# %% [markdown]
# # 02 · Language modeling and the bigram baseline
#
# **Goal:** turn raw text into tensors, define what a language model actually
# optimizes, and train the simplest one that works — a bigram model. It will be
# bad, and *how* it is bad is precisely the motivation for attention.
#
# **You will be able to:**
# 1. tokenize text and build `(x, y)` next-token training batches
# 2. state the training objective and read the loss in absolute terms
# 3. train a bigram language model in ~20 lines and sample from it
# 4. compute perplexity and explain what it means
# 5. articulate the bigram model's structural limitation
#
# Prerequisite: notebook 01.

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
print("device:", cu.describe_device(device))

# %% [markdown]
# ## 1 · The corpus
#
# `cu.get_text()` returns one long string. It tries `data/input.txt` first, then
# downloads tiny-shakespeare (~1.1 MB), and if there is no network it writes a
# deterministic synthetic play-script corpus so the notebook still runs. Drop
# any UTF-8 text file at `data/input.txt` to use your own.

# %%
text = cu.get_text()
print(f"{len(text):,} characters")
print("-" * 60)
print(text[:400])

# %% [markdown]
# > **A note on the numbers you will see.** Every loss in this course depends on
# > which corpus you ended up with. Tiny-shakespeare is genuinely hard (a good
# > small model reaches ~1.5 nats/char); the synthetic fallback is repetitive and
# > far easier (a small model can get under 0.5). Never compare a loss across
# > corpora — compare models trained on the *same* data, which is what each
# > notebook does.

# %% [markdown]
# ## 2 · Tokenization, the character-level version
#
# A model consumes integers, not characters. The mapping from text to integers
# is the **tokenizer**. The simplest choice — one token per unique character —
# gives a tiny vocabulary and no out-of-vocabulary problem, at the cost of very
# long sequences. Notebook 06 replaces it with byte-pair encoding.

# %%
tok = cu.CharTokenizer(text)
print(tok)
print("alphabet:", "".join(tok.chars).replace("\n", "\\n"))

sample = "The king speaks"
ids = tok.encode(sample)
print("encode:", ids)
print("decode:", repr(tok.decode(ids)))
assert tok.decode(tok.encode(sample)) == sample     # round-trip

# %% [markdown]
# ### The tokenizer trade-off, quantified

# %%
n_chars = len(text)
n_words = len(text.split())
print(f"characters : {n_chars:>9,}  vocab {tok.vocab_size:>6,}")
print(f"words      : {n_words:>9,}  vocab {len(set(text.split())):>6,}")
print(f"\nchar-level sequences are ~{n_chars / n_words:.1f}x longer than word-level,")
print("and attention cost grows with the SQUARE of sequence length. That tension")
print("is why real models use subword tokenization (notebook 06).")

# %% [markdown]
# ## 3 · Text as one long tensor, split into train and validation
#
# We hold back the last 10% to detect memorization. A model can always reduce
# training loss by memorizing; only validation loss tells you it learned
# something transferable.

# %%
data = torch.tensor(tok.encode(text), dtype=torch.long)
train_data, val_data = cu.train_val_split(data, frac=0.9)
splits = {"train": train_data, "val": val_data}
print("all:", tuple(data.shape), "| train:", tuple(train_data.shape),
      "| val:", tuple(val_data.shape))
print("first 20 ids:", data[:20].tolist())

# %% [markdown]
# ## 4 · The objective: predict the next token, at every position
#
# Given a window of `block_size` tokens, the target is the *same window shifted
# by one*. One window of length 8 therefore contains 8 training examples, with
# contexts of length 1, 2, ..., 8. This is why transformer training is so
# sample-efficient: every position supervises a prediction.

# %%
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    ctx = x[:t + 1]
    print(f"context {repr(tok.decode(ctx)):>22}  ->  target {tok.decode([y[t]])!r}")

# %% [markdown]
# ### Batches: many random windows at once
#
# `get_batch` samples `B` random start offsets and stacks the windows. Randomly
# sampling offsets every step is a cheap stand-in for shuffling a dataset.

# %%
def get_batch(split_data, batch_size, block_size, device="cpu"):
    ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
    x = torch.stack([split_data[i:i + block_size] for i in ix])         # (B, T)
    y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix]) # (B, T)
    return x.to(device), y.to(device)


xb, yb = get_batch(train_data, batch_size=4, block_size=8)
print("x:", tuple(xb.shape), " y:", tuple(yb.shape))
print(xb)
print("x row 0:", repr(tok.decode(xb[0])), "-> y row 0:", repr(tok.decode(yb[0])))
print("\n(y is x shifted left by one, so y[:, t] is the answer for x[:, :t+1])")

# %% [markdown]
# ## 5 · The bigram language model
#
# The whole model is a lookup table of shape `(V, V)`: row `i` holds the logits
# for "what token follows token `i`". `nn.Embedding` *is* that table, and
# indexing it is the forward pass. Note what is absent: no dependence on
# anything before the immediately preceding token.

# %%
class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # row i = logits over the whole vocabulary for "token after i"
        self.table = nn.Embedding(vocab_size, vocab_size)   # (V, V)

    def forward(self, idx, targets=None):
        logits = self.table(idx)                            # (B, T) -> (B, T, V)
        loss = None
        if targets is not None:
            V = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                     # (B, T, V)
            logits = logits[:, -1, :] / temperature   # (B, V) — only last step
            probs = F.softmax(logits, dim=-1)         # (B, V)
            nxt = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat([idx, nxt], dim=1)        # (B, T+1)
        return idx


bigram = BigramLM(tok.vocab_size)
print(bigram)
print("parameters:", cu.human(cu.count_params(bigram)))

# %% [markdown]
# ### Reading the loss before training
#
# An untrained model knows nothing, so it should be about as good as guessing
# uniformly: loss ~ `ln(V)`. **Always check this before training anything.** If
# your untrained loss is wildly off, your loss wiring or your init is wrong.
#
# It comes out slightly *above* `ln(V)` here because `nn.Embedding` initializes
# from N(0, 1): the rows are not flat, they are confidently wrong, and confident
# wrong answers cost more than uniform ones. Scaling the table down at init
# (`* 0.01`) lands almost exactly on `ln(V)` — try it.

# %%
xb, yb = get_batch(train_data, 32, block_size)
_, loss0 = bigram(xb, yb)
print(f"initial loss  : {loss0.item():.4f}")
print(f"ln(V) = ln({tok.vocab_size}) : {math.log(tok.vocab_size):.4f}")

# %% [markdown]
# ### Untrained sample: pure noise

# %%
start = torch.zeros((1, 1), dtype=torch.long)
print(repr(tok.decode(bigram.generate(start, 150)[0])))

# %% [markdown]
# ## 6 · Train it

# %%
bigram = BigramLM(tok.vocab_size).to(device)
opt = torch.optim.AdamW(bigram.parameters(), lr=1e-2)

history = {"step": [], "train": [], "val": [], "step_loss": []}
steps = 3000
for step in range(steps + 1):
    if step % 500 == 0:
        ev = cu.estimate_loss(bigram, splits, 64, block_size, device, iters=30)
        history["step"].append(step)
        history["train"].append(ev["train"])
        history["val"].append(ev["val"])
        print(f"step {step:5d} | train {ev['train']:.4f} | val {ev['val']:.4f}")
    if step == steps:
        break
    xb, yb = get_batch(train_data, 64, block_size, device)
    _, loss = bigram(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    history["step_loss"].append(loss.item())

# %%
try:
    import matplotlib.pyplot as plt
    cu.plot_history(history, "bigram model")
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %% [markdown]
# ## 7 · Interpreting the number
#
# Loss is average `-log p(correct token)`, in nats. Two useful conversions:
#
# - **bits per character** = loss / ln(2) — the compression rate the model achieves.
# - **perplexity** = exp(loss) — "effectively how many equally-likely choices is
#   the model picking between at each step".
#
# A uniform model over 47 characters has perplexity 47. Getting to ~10 means the
# model has genuinely narrowed the field, even with no notion of context.

# %%
final = history["val"][-1]
print(f"val loss      : {final:.4f} nats")
print(f"bits per char : {final / math.log(2):.4f}")
print(f"perplexity    : {math.exp(final):.2f}  (uniform baseline: {tok.vocab_size})")

# %% [markdown]
# ### What the model learned, visualized
#
# The learned table is a next-character transition matrix. Reading a row tells
# you the model's entire belief given that character.

# %%
probs = F.softmax(bigram.table.weight.detach().cpu().float(), dim=-1)   # (V, V)
for ch in ["t", "h", "q", " "]:
    if ch not in tok.stoi:
        continue
    row = probs[tok.stoi[ch]]
    top = torch.topk(row, 5)
    pretty = ", ".join(f"{tok.itos[int(i)]!r}:{p:.2f}"
                       for p, i in zip(top.values, top.indices))
    print(f"after {ch!r:>4} -> {pretty}")

# %%
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 6))
    labels = [c.replace("\n", "\\n") for c in tok.chars]
    cu.show_attention(probs, tokens=labels, ax=ax,
                      title="bigram transition probabilities")
    ax.set_xlabel("next character"); ax.set_ylabel("current character")
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %% [markdown]
# ## 8 · Sample from the trained model

# %%
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(repr(tok.decode(bigram.generate(start, 300, temperature=1.0)[0])))

# %% [markdown]
# Character *pairs* look plausible ("th", "he", " t"), and words are the right
# length — but nothing survives beyond two characters. That is not a training
# failure; it is the model class. `p(next | previous one token)` is all the
# architecture can express.

# %% [markdown]
# ## 9 · Temperature: the first decoding knob
#
# Dividing logits by `T` before softmax reshapes the distribution:
# `T < 1` sharpens it (safer, more repetitive), `T > 1` flattens it (more
# surprising, more garbage). Notebook 12 covers top-k, nucleus sampling and
# beam search.

# %%
for temperature in (0.3, 1.0, 1.8):
    out = bigram.generate(start, 120, temperature=temperature)[0]
    print(f"--- T={temperature} ---")
    print(repr(tok.decode(out)))

# %% [markdown]
# ## 10 · Why we need something better
#
# Extend the idea and the problem becomes obvious. An n-gram table needs `V**n`
# parameters, so context length costs *exponentially*:

# %%
V = tok.vocab_size
for n in range(2, 8):
    print(f"{n}-gram table: V**{n} = {V ** n:>22,} parameters "
          f"({cu.human(V ** n)})")
print("\nA transformer with a 256-token context uses ~10M parameters total.")

# %% [markdown]
# Concretely, here is a prediction no bigram can ever make. In the string
# `"the king ... the k"`, the right continuation depends on tokens *far* back,
# but the bigram sees only `"k"`. We need a model where a prediction can depend
# on **all** previous tokens, with a parameter count that grows gently — and
# where which earlier tokens matter is *learned*, not fixed.
#
# That mechanism is attention.

# %%
ctx = "the k"
idx = torch.tensor([tok.encode(ctx)], device=device)
logits, _ = bigram(idx)
top = torch.topk(F.softmax(logits[0, -1], dim=-1), 4)
print(f"bigram, given {ctx!r} (it only looks at 'k'):")
print("  ", ", ".join(f"{tok.itos[int(i)]!r}:{p:.2f}"
                      for p, i in zip(top.values, top.indices)))

# %% [markdown]
# ## Exercises
#
# 1. **Count, don't learn.** Build the bigram probability table by *counting*
#    transitions in `train_data` (add-one smoothing), and compare its validation
#    loss to the trained model's. Why are they so close?
# 2. **Block size sweep.** Train the bigram at `block_size` 4, 8, 64. Explain why
#    the loss barely changes, unlike what you'd expect from a transformer.
# 3. **Trigram.** Implement a trigram model by embedding the *pair* of previous
#    tokens (`idx[:, :-1] * V + idx[:, 1:]` as the lookup index). How much does
#    validation loss improve, and how many parameters did it cost?
# 4. **Perplexity by hand.** Write `perplexity(model, data)` that iterates over
#    non-overlapping windows of the validation set and returns `exp(mean loss)`.
# 5. **Your own corpus.** Put a text file at `data/input.txt`, re-run, and
#    compare `ln(V)` against the final loss for your alphabet.
#
# ---
# **Next:** [03 · Attention from scratch](03_attention_from_scratch.ipynb) — build
# self-attention one tensor at a time, starting from a plain average.
