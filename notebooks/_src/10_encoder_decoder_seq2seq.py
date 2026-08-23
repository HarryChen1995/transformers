# %% [markdown]
# # 10 · Encoder-decoder transformers and seq2seq
#
# **Goal:** build the *full* architecture from "Attention Is All You Need" — a
# bidirectional encoder, a causal decoder, and cross-attention between them —
# and train it on a real translation-shaped task where we can *see* the
# alignment it learns.
#
# **Task:** normalize messy dates to ISO format.
# `"3rd of March, 2019" -> "2019-03-03"`. Small enough to train in a couple of
# minutes, structured enough that cross-attention learns a legible alignment.
#
# **You will be able to:**
# 1. explain when to choose encoder-decoder over decoder-only
# 2. implement cross-attention, and say exactly which tensor comes from where
# 3. build padding masks and combine them with causal masks correctly
# 4. train with teacher forcing and decode greedily at inference
# 5. read a cross-attention map as an alignment matrix
#
# Prerequisites: notebooks 01–05 (07–08 optional).

# %%
import math
import random
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
# ## 1 · Three transformer families
#
# | family | attention | trained to | good at | examples |
# |---|---|---|---|---|
# | **decoder-only** | causal self | predict next token | generation, everything | GPT, LLaMA |
# | **encoder-only** | bidirectional self | fill in masked tokens | classification, embedding | BERT (nb 11) |
# | **encoder-decoder** | both + cross | map sequence A to sequence B | translation, summarization | T5, Whisper, NMT |
#
# The encoder-decoder split makes sense when the input is *given in full* and the
# output is a different sequence: the encoder can read the input bidirectionally
# (nothing to hide), while the decoder generates left to right.
#
# Decoder-only models handle the same tasks by concatenating input and output
# into one stream, which is simpler and scales better — which is why decoder-only
# won for general LLMs. Encoder-decoder still dominates where inputs and outputs
# are genuinely different modalities or lengths (speech → text).

# %% [markdown]
# ## 2 · The dataset
#
# We generate pairs on the fly, so we get unlimited data with controlled
# difficulty and there is no download.

# %%
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
ORD = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}


def make_pair(rng: random.Random) -> tuple[str, str]:
    y = rng.randint(1970, 2029)
    mo = rng.randint(1, 12)
    d = rng.randint(1, 28)
    name = MONTHS[mo - 1]
    suffix = ORD.get(d, "th")
    style = rng.randint(0, 5)
    if style == 0:
        src = f"{d} {name} {y}"
    elif style == 1:
        src = f"{name} {d}, {y}"
    elif style == 2:
        src = f"{d}{suffix} of {name.capitalize()}, {y}"
    elif style == 3:
        src = f"{name[:3]} {d} {y}"
    elif style == 4:
        src = f"{d:02d}/{mo:02d}/{y}"
    else:
        src = f"{y}, {name} {d}{suffix}"
    return src, f"{y}-{mo:02d}-{d:02d}"


rng = random.Random(0)
for _ in range(8):
    src, tgt = make_pair(rng)
    print(f"  {src:<32} -> {tgt}")

# %% [markdown]
# The task is genuinely non-trivial: the model must *reorder* (year moves to the
# front), *translate* (month name to number), *pad* (`3` to `03`), and *copy*
# (digits of the year) — all from characters, with no hand-written rules.

# %% [markdown]
# ## 3 · Vocabulary and special tokens
#
# Seq2seq needs three special tokens:
#
# - `<pad>` — fills short sequences so a batch is rectangular. **Must be masked**
#   so attention ignores it and the loss doesn't score it.
# - `<bos>` — the decoder's first input, so it has something to condition on.
# - `<eos>` — how the model says "done" at inference time.

# %%
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


class Seq2SeqVocab:
    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.itos = [PAD, BOS, EOS] + chars
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2

    def encode(self, s, bos=False, eos=False):
        ids = [self.stoi[c] for c in s]
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, strip_special=True):
        out = []
        for i in ids:
            i = int(i)
            if strip_special and i in (self.pad_id, self.bos_id):
                continue
            if strip_special and i == self.eos_id:
                break
            out.append(self.itos[i])
        return "".join(out)

    def __len__(self):
        return len(self.itos)


rng = random.Random(1)
pairs = [make_pair(rng) for _ in range(20000)]
vocab = Seq2SeqVocab([s for p in pairs for s in p])
SRC_LEN = max(len(s) for s, _ in pairs) + 1
TGT_LEN = 10 + 2                                   # "YYYY-MM-DD" + bos + eos
print(f"vocab {len(vocab)}: {vocab.itos}")
print(f"src max len {SRC_LEN} | tgt len {TGT_LEN}")

split = int(0.9 * len(pairs))
train_pairs, val_pairs = pairs[:split], pairs[split:]
print(f"train {len(train_pairs):,} | val {len(val_pairs):,} pairs")


def make_batch(pair_list, batch_size, rng, device="cpu"):
    chosen = [pair_list[rng.randrange(len(pair_list))] for _ in range(batch_size)]
    src = torch.full((batch_size, SRC_LEN), vocab.pad_id, dtype=torch.long)
    tgt_in = torch.full((batch_size, TGT_LEN - 1), vocab.pad_id, dtype=torch.long)
    tgt_out = torch.full((batch_size, TGT_LEN - 1), vocab.pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(chosen):
        s_ids = vocab.encode(s)
        src[i, :len(s_ids)] = torch.tensor(s_ids)
        full = vocab.encode(t, bos=True, eos=True)      # <bos> Y Y Y Y - ... <eos>
        tgt_in[i, :len(full) - 1] = torch.tensor(full[:-1])
        tgt_out[i, :len(full) - 1] = torch.tensor(full[1:])
    return src.to(device), tgt_in.to(device), tgt_out.to(device)


src, tgt_in, tgt_out = make_batch(train_pairs, 4, random.Random(0))
print("\nsrc   :", tuple(src.shape))
print("tgt_in:", tuple(tgt_in.shape), "(starts with <bos>)")
print("tgt_out:", tuple(tgt_out.shape), "(shifted by one, ends with <eos>)")
print("\nrow 0:")
print("  src    ", repr(vocab.decode(src[0])))
print("  tgt_in ", repr(vocab.decode(tgt_in[0], strip_special=False)))
print("  tgt_out", repr(vocab.decode(tgt_out[0], strip_special=False)))

# %% [markdown]
# ## 4 · Masks: the part everyone gets wrong
#
# Three masks, three jobs:
#
# 1. **encoder padding mask** `(B, 1, 1, S)` — encoder self-attention must ignore
#    `<pad>` keys.
# 2. **decoder causal + padding mask** `(B, 1, T, T)` — no peeking ahead, and no
#    attending to padded targets.
# 3. **cross-attention padding mask** `(B, 1, 1, S)` — decoder queries must not
#    attend to padded source positions.
#
# Forgetting #1 or #3 lets the model attend to garbage; forgetting #2 lets it
# cheat. All three are boolean tensors that broadcast against `(B, h, T, S)`.

# %%
def padding_mask(ids, pad_id: int):
    """(B, S) -> (B, 1, 1, S) True where a key is REAL."""
    return (ids != pad_id).unsqueeze(1).unsqueeze(2)


def causal_mask(T: int, device):
    """(1, 1, T, T) True where a query may attend."""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril().view(1, 1, T, T)


src_mask = padding_mask(src, vocab.pad_id)
tgt_pad = padding_mask(tgt_in, vocab.pad_id)
tgt_mask = tgt_pad & causal_mask(tgt_in.size(1), tgt_in.device)
print("src_mask:", tuple(src_mask.shape), "| tgt_mask:", tuple(tgt_mask.shape))
print("\ndecoder mask for row 0 (True = allowed):")
print(tgt_mask[0, 0].int())

# %% [markdown]
# ## 5 · Attention that handles both self and cross
#
# One module covers every case. The only structural difference between self- and
# cross-attention is where K and V come from:
#
# ```
# self-attention : Q, K, V  all from x
# cross-attention: Q from the decoder, K and V from the ENCODER output
# ```
#
# That's it. Cross-attention is not a new mechanism — it is the same four lines
# with a different argument.

# %%
class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key_value=None, mask=None, return_weights=False):
        kv = query if key_value is None else key_value       # self vs cross
        B, T, C = query.shape
        S = kv.size(1)
        q = self.q_proj(query).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.k_proj(kv).view(B, S, self.h, self.d).transpose(1, 2)
        v = self.v_proj(kv).view(B, S, self.h, self.d).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.d ** -0.5      # (B, h, T, S)
        if mask is not None:
            att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = self.drop(att) @ v                                # (B, h, T, d)
        y = self.out(y.transpose(1, 2).contiguous().view(B, T, C))
        return (y, att) if return_weights else y


# quick shape check for both modes
a = Attention(64, 4)
dec_x, enc_out = torch.randn(2, 5, 64), torch.randn(2, 11, 64)
print("self :", tuple(a(dec_x).shape))
_, w = a(dec_x, enc_out, return_weights=True)
print("cross:", tuple(a(dec_x, enc_out).shape), "| weights", tuple(w.shape),
      "= (B, h, T_decoder, S_encoder)")

# %% [markdown]
# ## 6 · The full model

# %%
@dataclass
class S2SConfig:
    vocab_size: int = 40
    d_model: int = 128
    n_heads: int = 4
    n_enc_layers: int = 3
    n_dec_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 64


class FF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Bidirectional self-attention + FF. No causal mask — the encoder sees all."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ff = FF(cfg)

    def forward(self, x, src_mask):
        x = x + self.attn(self.ln1(x), mask=src_mask)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderLayer(nn.Module):
    """Three sublayers: causal self-attn, cross-attn to the encoder, then FF."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.ff = FF(cfg)

    def forward(self, x, enc_out, tgt_mask, src_mask, return_weights=False):
        x = x + self.self_attn(self.ln1(x), mask=tgt_mask)
        if return_weights:
            y, w = self.cross_attn(self.ln2(x), enc_out, mask=src_mask,
                                   return_weights=True)
        else:
            y, w = self.cross_attn(self.ln2(x), enc_out, mask=src_mask), None
        x = x + y
        x = x + self.ff(self.ln3(x))
        return (x, w) if return_weights else x


class Transformer(nn.Module):
    def __init__(self, cfg: S2SConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_enc_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_dec_layers)])
        self.ln_enc, self.ln_dec = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def embed(self, ids):
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device)
        return self.drop(self.tok_emb(ids) + self.pos_emb(pos))

    def encode(self, src, src_mask):
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.ln_enc(x)                                  # (B, S, C)

    def decode(self, tgt_in, enc_out, tgt_mask, src_mask, return_weights=False):
        x = self.embed(tgt_in)
        weights = []
        for layer in self.decoder:
            if return_weights:
                x, w = layer(x, enc_out, tgt_mask, src_mask, return_weights=True)
                weights.append(w)
            else:
                x = layer(x, enc_out, tgt_mask, src_mask)
        logits = self.head(self.ln_dec(x))                     # (B, T, V)
        return (logits, weights) if return_weights else logits

    def forward(self, src, tgt_in, tgt_out=None, return_weights=False):
        src_mask = padding_mask(src, vocab.pad_id)
        tgt_mask = (padding_mask(tgt_in, vocab.pad_id)
                    & causal_mask(tgt_in.size(1), tgt_in.device))
        enc_out = self.encode(src, src_mask)
        res = self.decode(tgt_in, enc_out, tgt_mask, src_mask, return_weights)
        logits, weights = res if return_weights else (res, None)

        loss = None
        if tgt_out is not None:
            # ignore_index skips <pad> positions so padding never affects the loss
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   tgt_out.reshape(-1),
                                   ignore_index=vocab.pad_id)
        return (logits, loss, weights) if return_weights else (logits, loss)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=12, return_weights=False):
        self.eval()
        src_mask = padding_mask(src, vocab.pad_id)
        enc_out = self.encode(src, src_mask)                   # encode ONCE
        B = src.size(0)
        ys = torch.full((B, 1), vocab.bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        weights = None
        for _ in range(max_len):
            tgt_mask = causal_mask(ys.size(1), src.device)
            out = self.decode(ys, enc_out, tgt_mask, src_mask, return_weights)
            logits, weights = out if return_weights else (out, None)
            nxt = logits[:, -1].argmax(-1, keepdim=True)        # greedy
            nxt = torch.where(finished.unsqueeze(1),
                              torch.full_like(nxt, vocab.pad_id), nxt)
            ys = torch.cat([ys, nxt], dim=1)
            finished |= nxt.squeeze(1) == vocab.eos_id
            if finished.all():
                break
        return (ys, weights) if return_weights else ys


cfg = S2SConfig(vocab_size=len(vocab), max_len=max(SRC_LEN, TGT_LEN) + 2)
model = Transformer(cfg).to(device)
print("params:", cu.human(cu.count_params(model)))
print("encoder layers:", len(model.encoder), "| decoder layers:", len(model.decoder))

# %% [markdown]
# ### Teacher forcing
#
# During training the decoder is fed the **correct** previous tokens (`tgt_in`),
# not its own predictions — so all `T` positions train in parallel, exactly like
# a decoder-only LM. At inference it must consume its own output, which creates
# *exposure bias*: one early mistake can derail the rest. Beam search (notebook
# 12) partly compensates.

# %%
_, loss = model(src.to(device), tgt_in.to(device), tgt_out.to(device))
print(f"initial loss {loss.item():.4f} vs ln(V) = {math.log(len(vocab)):.4f}")

# %% [markdown]
# ## 7 · Train

# %%
def exact_match(model, pair_list, n=300, rng=None):
    """The metric that matters: is the whole output string right?"""
    rng = rng or random.Random(0)
    chosen = [pair_list[rng.randrange(len(pair_list))] for _ in range(n)]
    src_b = torch.full((n, SRC_LEN), vocab.pad_id, dtype=torch.long)
    for i, (s, _) in enumerate(chosen):
        ids = vocab.encode(s)
        src_b[i, :len(ids)] = torch.tensor(ids)
    out = model.greedy_decode(src_b.to(device))
    preds = [vocab.decode(row) for row in out]
    correct = sum(p == t for p, (_, t) in zip(preds, chosen))
    return correct / n, list(zip((s for s, _ in chosen), preds,
                                 (t for _, t in chosen)))[:5]


cu.set_seed(1337)
model = Transformer(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01,
                        betas=(0.9, 0.98))
STEPS, BATCH, WARMUP = 3000, 64, 200
data_rng = random.Random(0)
history = {"step": [], "train": [], "val": [], "acc": [], "step_loss": []}

for step in range(STEPS + 1):
    lr = 3e-4 * min(1.0, (step + 1) / WARMUP) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / STEPS))))
    for g in opt.param_groups:
        g["lr"] = lr

    if step % 500 == 0 or step == STEPS:
        model.eval()
        with torch.no_grad():
            vs, vi, vo = make_batch(val_pairs, 256, random.Random(7), device)
            _, vloss = model(vs, vi, vo)
            ts, ti, to = make_batch(train_pairs, 256, random.Random(8), device)
            _, tloss = model(ts, ti, to)
        acc, samples = exact_match(model, val_pairs, n=200)
        history["step"].append(step)
        history["train"].append(tloss.item())
        history["val"].append(vloss.item())
        history["acc"].append(acc)
        print(f"step {step:5d} | train {tloss.item():.4f} | val {vloss.item():.4f} "
              f"| exact match {acc:6.1%}")
        model.train()
    if step == STEPS:
        break

    s, ti, to = make_batch(train_pairs, BATCH, data_rng, device)
    _, loss = model(s, ti, to)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    history["step_loss"].append(loss.item())

# %%
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(history["step_loss"], alpha=0.3, lw=0.7)
    axes[0].plot(history["step"], history["train"], "o-", label="train")
    axes[0].plot(history["step"], history["val"], "s-", label="val")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["step"], [100 * a for a in history["acc"]], "o-", color="green")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("exact match %")
    axes[1].set_ylim(0, 101); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %% [markdown]
# ### Predictions

# %%
acc, samples = exact_match(model, val_pairs, n=500, rng=random.Random(99))
print(f"exact-match accuracy on 500 unseen dates: {acc:.1%}\n")
for s, pred, gold in samples:
    flag = "ok " if pred == gold else "BAD"
    print(f"  {flag} {s:<30} -> {pred:<12} (gold {gold})")

print("\nhand-written inputs, including formats the training data never used:")
for s in ["1 january 1999", "december 25, 2020", "07/04/1976",
          "23rd of August, 2011", "2 feb 2000"]:
    ids = vocab.encode(s)
    b = torch.full((1, SRC_LEN), vocab.pad_id, dtype=torch.long)
    b[0, :len(ids)] = torch.tensor(ids)
    out = model.greedy_decode(b.to(device))
    print(f"  {s:<26} -> {vocab.decode(out[0])}")

# %% [markdown]
# ## 8 · Read the cross-attention as an alignment
#
# This is the payoff. The rows are output characters, the columns are input
# characters. A model that solved the task properly should show the year digits
# of the output attending to the year digits of the input, the month digits
# attending to the month *word*, and so on — an alignment matrix it was never
# explicitly taught.

# %%
probe = "3rd of March, 2019"
ids = vocab.encode(probe)
b = torch.full((1, SRC_LEN), vocab.pad_id, dtype=torch.long)
b[0, :len(ids)] = torch.tensor(ids)
out, weights = model.greedy_decode(b.to(device), return_weights=True)
pred = vocab.decode(out[0])
print(f"{probe!r} -> {pred!r}")
print("cross-attention per decoder layer:", [tuple(w.shape) for w in weights])

# %%
try:
    import matplotlib.pyplot as plt
    W = weights[-1][0].mean(0).cpu()                 # last layer, mean over heads
    out_chars = [vocab.itos[int(i)] for i in out[0]][1:]
    src_chars = list(probe)
    W = W[:len(out_chars), :len(src_chars)]
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(W, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(src_chars)), src_chars, fontsize=9)
    ax.set_yticks(range(len(out_chars)), out_chars, fontsize=9)
    ax.set_xlabel("source (input date)"); ax.set_ylabel("generated (ISO date)")
    ax.set_title("cross-attention alignment, last decoder layer (mean over heads)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %%
W = weights[-1][0].mean(0).cpu()
out_chars = [vocab.itos[int(i)] for i in out[0]][1:]
print("for each generated character, which input character it looked at most:")
for i, ch in enumerate(out_chars):
    if ch in (PAD, EOS):
        continue
    j = int(W[i, :len(probe)].argmax())
    print(f"  output {ch!r:>6} <- input position {j:>2} {probe[j]!r} "
          f"(weight {W[i, j]:.2f})")

# %% [markdown]
# ## 9 · Encoder-decoder vs decoder-only on the same task
#
# A decoder-only model can do this too: feed it `"3 march 2019 = 2019-03-03"` as
# one stream and mask the loss to the part after `=`. Let's confirm, and note the
# trade-offs.

# %%
from tinygpt import GPTConfig, TinyGPT

SEP = "="
flat_texts = [f"{s}{SEP}{t}\n" for s, t in train_pairs[:8000]]
flat = "".join(flat_texts)
flat_tok = cu.CharTokenizer(flat)
flat_ids = torch.tensor(flat_tok.encode(flat), dtype=torch.long)
tr, va = cu.train_val_split(flat_ids)

cu.set_seed(1337)
dec_only = TinyGPT(GPTConfig(vocab_size=flat_tok.vocab_size, block_size=64,
                             d_model=128, n_heads=4, n_layers=4, dropout=0.1))
print(f"decoder-only ({cu.human(cu.count_params(dec_only))} params) on the flat format:")
cu.train(dec_only, {"train": tr, "val": va}, steps=1500, batch_size=64,
         block_size=64, lr=1e-3, device=device, eval_every=750)

correct = 0
trials = 200
probe_rng = random.Random(123)
for _ in range(trials):
    s, gold = val_pairs[probe_rng.randrange(len(val_pairs))]
    ctx = torch.tensor([flat_tok.encode(f"{s}{SEP}")], device=device)
    out = dec_only.generate(ctx, 11, temperature=0.01)      # ~greedy
    got = flat_tok.decode(out[0])[len(s) + 1:].split("\n")[0][:10]
    correct += got == gold
print(f"\ndecoder-only exact match: {correct / trials:.1%}")
print(f"encoder-decoder exact match: {acc:.1%}")
print("""
Trade-offs, at this scale and beyond:
  encoder-decoder : bidirectional view of the input; cross-attention gives a
                    clean alignment; encoder runs once per input at inference.
  decoder-only    : one stack, one objective, trivially scalable, and it can be
                    prompted for other tasks. Pays O((S+T)^2) attention over the
                    concatenation instead of O(S^2) + O(T^2) + O(S*T).
""")

# %% [markdown]
# ## Exercises
#
# 1. **Break a mask.** Remove `src_mask` from cross-attention and retrain. The
#    model now attends to `<pad>`; measure the accuracy cost.
# 2. **Remove `ignore_index`.** Let the loss score `<pad>` targets. Explain why
#    the loss looks *better* while accuracy gets worse.
# 3. **Harder inputs.** Add noisy formats (typos, `"the 3rd of Mar '19"`,
#    two-digit years). Where does exact match break down?
# 4. **Depth split.** Compare 3+3 encoder/decoder layers against 5+1 and 1+5 at
#    matched parameters. Which half of the model does the work here?
# 5. **Length generalization.** Train on years 1970-2019 only, then test on
#    2020-2029. Does the model generalize, or has it memorized year prefixes?
#
# ---
# **Next:** [11 · Encoder-only models: BERT-style pretraining and fine-tuning](11_bert_and_finetuning.ipynb)
