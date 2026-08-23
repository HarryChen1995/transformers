# %% [markdown]
# # 11 · Encoder-only models: BERT-style pretraining and fine-tuning
#
# **Goal:** build the third family — a bidirectional encoder — pretrain it with
# masked language modeling, then fine-tune it for classification. Along the way,
# implement the three fine-tuning strategies you will actually choose between in
# practice: full fine-tuning, head-only (linear probe), and LoRA.
#
# **You will be able to:**
# 1. explain why MLM exists and how the 80/10/10 masking rule works
# 2. pretrain a small encoder and watch it learn to fill in blanks
# 3. attach a `[CLS]` classification head and fine-tune it
# 4. implement LoRA from scratch and count exactly what it saves
# 5. choose between the strategies from data size and compute budget
#
# Prerequisites: notebooks 01–05, 10 (for masks).

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
# ## 1 · Why bidirectional needs a different objective
#
# A causal LM trains on "predict the next token" — which only works because the
# future is hidden. Remove the causal mask and that objective becomes trivial:
# the answer is sitting in the input.
#
# So BERT changes the *task* instead: corrupt some tokens, and predict what they
# were, using context from **both** sides. That gives every token a
# representation informed by the whole sequence, which is what you want for
# classification, retrieval and embeddings — and useless for generation.

# %%
print("""
causal LM (GPT)              masked LM (BERT)
---------------              ----------------
the king [?]                 the [MASK] speaks
  ^ predict from left only      ^ predict from BOTH sides

Consequences:
  * BERT can't generate text autoregressively — it has no left-to-right factorization
  * BERT trains on only ~15% of positions per pass (the masked ones), so it needs
    more passes over the data than a causal LM, which trains on 100% of positions
  * BERT's representations are better for classification/embedding at equal size
""")

# %% [markdown]
# ## 2 · A dataset designed to show transfer
#
# To see what pretraining actually buys, the task has to be one that labels alone
# cannot solve. So we build the classic transfer scenario:
#
# - **Unlabelled pretraining corpus** — sentences pairing an adjective with a
#   *consequence clause* that correlates with its polarity
#   (`"the play is fair and i shall return"`). No labels anywhere.
# - **Labelled fine-tuning set** — short sentences with **no** clause
#   (`"the king is lovely"`), using only *four* positive and four negative
#   adjectives.
# - **Validation set** — same short form, but using the **eight held-out
#   adjectives the classifier has never seen a label for**.
#
# A model trained from scratch on the labels can only memorize the adjectives it
# was shown, so it must fail on the held-out ones. A pretrained model has already
# placed those adjectives near their polarity-mates in embedding space — because
# they appeared with the same consequence clauses in unlabelled text — so its
# classifier generalizes.
#
# This is exactly why pretraining works on real text, in miniature.

# %%
POS_ADJ = ["good", "great", "lovely", "kind", "noble", "true", "fair", "sweet",
           "gentle", "wise", "merry", "bright"]
NEG_ADJ = ["bad", "vile", "cruel", "false", "bitter", "harsh", "poor", "grim",
           "dull", "weary", "foul", "idle"]
POS_TAIL = ["i shall return", "praise it well", "worth the coin",
            "my heart is light", "i stayed till dawn", "long may it stand"]
NEG_TAIL = ["i left at once", "curse the hour", "waste of coin",
            "my heart is heavy", "i slept throughout", "let it fall"]
SUBJ = ["the play", "this book", "the king", "my lord", "the letter", "her song",
        "the feast", "that tale"]

SEEN_POS, HELD_POS = POS_ADJ[:4], POS_ADJ[4:]
SEEN_NEG, HELD_NEG = NEG_ADJ[:4], NEG_ADJ[4:]


def pretrain_sentence(rng: random.Random) -> str:
    """Unlabelled: adjective + a clause that correlates with its polarity."""
    positive = rng.randint(0, 1)
    adj = rng.choice(POS_ADJ if positive else NEG_ADJ)
    tail = rng.choice(POS_TAIL if positive else NEG_TAIL)
    return f"{rng.choice(SUBJ)} is {adj} and {tail}"


def label_sentence(rng, pos_pool, neg_pool) -> tuple[str, int]:
    """Labelled: the bare adjective, no clause to lean on."""
    label = rng.randint(0, 1)
    adj = rng.choice(pos_pool if label else neg_pool)
    return f"{rng.choice(SUBJ)} is {adj}", label


rng = random.Random(0)
pre_texts = [pretrain_sentence(rng) for _ in range(6000)]
train_lab = [label_sentence(rng, SEEN_POS, SEEN_NEG) for _ in range(400)]
val_lab = [label_sentence(rng, HELD_POS, HELD_NEG) for _ in range(1000)]

print("unlabelled pretraining corpus (6,000 sentences):")
for t in pre_texts[:4]:
    print("   ", t)
print()
print(f"labelled train, adjectives {SEEN_POS + SEEN_NEG}:")
for t, y in train_lab[:3]:
    print(f"    [{'pos' if y else 'neg'}] {t}")
print()
print(f"validation, HELD-OUT adjectives {HELD_POS + HELD_NEG}:")
for t, y in val_lab[:3]:
    print(f"    [{'pos' if y else 'neg'}] {t}")

# %% [markdown]
# ## 3 · Vocabulary with `[CLS]` and `[MASK]`
#
# Word-level here, for legibility — the masking mechanics are identical at
# subword level. Four special tokens:
#
# - `[PAD]` — batch padding, always masked out
# - `[CLS]` — prepended to every sequence; its final hidden state can serve as
#   the sentence representation
# - `[MASK]` — the corruption token
# - `[UNK]` — unseen words

# %%
class WordVocab:
    def __init__(self, texts, min_count=1):
        from collections import Counter
        counts = Counter(w for t in texts for w in t.split())
        words = sorted(w for w, c in counts.items() if c >= min_count)
        self.itos = ["[PAD]", "[CLS]", "[MASK]", "[UNK]"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.cls_id, self.mask_id, self.unk_id = 0, 1, 2, 3

    def encode(self, s, cls=True, max_len=None):
        ids = [self.stoi.get(w, self.unk_id) for w in s.split()]
        if cls:
            ids = [self.cls_id] + ids
        if max_len:
            ids = ids[:max_len] + [self.pad_id] * max(0, max_len - len(ids))
        return ids

    def decode(self, ids):
        return " ".join(self.itos[int(i)] for i in ids)

    def __len__(self):
        return len(self.itos)


all_texts = pre_texts + [t for t, _ in train_lab + val_lab]
vocab = WordVocab(all_texts)
MAX_LEN = max(len(t.split()) for t in all_texts) + 1
print(f"vocab {len(vocab)} | max_len {MAX_LEN}")
print("specials:", vocab.itos[:4])
print("encoded:", vocab.encode(pre_texts[0], max_len=MAX_LEN))
print("decoded:", vocab.decode(vocab.encode(pre_texts[0], max_len=MAX_LEN)))

# %% [markdown]
# ## 4 · The encoder
#
# Structurally identical to notebook 05's stack with one change: **no causal
# mask**. Every token attends to every non-padding token.

# %%
@dataclass
class EncoderConfig:
    vocab_size: int = 100
    max_len: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1


class BidirectionalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.d = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, pad_mask=None, return_weights=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = lambda t: t.view(B, T, self.h, self.d).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        att = (q @ k.transpose(-2, -1)) * self.d ** -0.5      # (B, h, T, T)
        if pad_mask is not None:
            att = att.masked_fill(~pad_mask, float("-inf"))   # NO causal mask
        att = F.softmax(att, dim=-1)
        y = self.drop(att) @ v
        y = self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        return (y, att) if return_weights else y


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = BidirectionalAttention(cfg)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                                nn.Linear(cfg.d_ff, cfg.d_model),
                                nn.Dropout(cfg.dropout))

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.ff(self.ln2(x))
        return x


class Encoder(nn.Module):
    """A miniature BERT: embeddings + bidirectional blocks + two heads."""

    def __init__(self, cfg: EncoderConfig, n_classes: int = 2):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.cls_head = nn.Linear(cfg.d_model, n_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, ids, return_weights=False):
        B, T = ids.shape
        pad_mask = (ids != 0).unsqueeze(1).unsqueeze(2)          # (B,1,1,T)
        pos = torch.arange(T, device=ids.device)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        weights = []
        for block in self.blocks:
            if return_weights:
                y, w = block.attn(block.ln1(x), pad_mask, return_weights=True)
                x = x + y
                x = x + block.ff(block.ln2(x))
                weights.append(w)
            else:
                x = block(x, pad_mask)
        x = self.ln_f(x)
        return (x, weights) if return_weights else x

    def forward_mlm(self, ids, labels=None):
        x = self.encode(ids)
        logits = self.mlm_head(x)                                # (B, T, V)
        loss = None
        if labels is not None:
            # -100 marks "not masked, don't score"
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   labels.view(-1), ignore_index=-100)
        return logits, loss

    def forward_cls(self, ids, labels=None, pool="cls"):
        x = self.encode(ids)
        if pool == "cls":
            summary = x[:, 0]                                     # [CLS] position
        else:
            keep = (ids != 0).unsqueeze(-1).float()               # mean over real
            summary = (x * keep).sum(1) / keep.sum(1).clamp_min(1)
        logits = self.cls_head(summary)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return logits, loss


ecfg = EncoderConfig(vocab_size=len(vocab), max_len=MAX_LEN)
enc = Encoder(ecfg)
print("params:", cu.human(cu.count_params(enc)))

# %% [markdown]
# ## 5 · Masked language modeling, with the 80/10/10 rule
#
# BERT picks 15% of tokens and, of those:
#
# - **80%** are replaced with `[MASK]`
# - **10%** are replaced with a *random* token
# - **10%** are left **unchanged**
#
# Why the odd recipe? `[MASK]` never appears at fine-tuning time, so a model
# trained only on `[MASK]` sees a different input distribution downstream. The
# random and unchanged cases force it to build a good representation of *every*
# position, not just the marked ones.

# %%
def mask_tokens(ids: torch.Tensor, vocab, prob=0.15):
    """Returns (corrupted_ids, labels) with -100 where nothing is scored."""
    labels = ids.clone()
    special = (ids == vocab.pad_id) | (ids == vocab.cls_id)
    selected = (torch.rand(ids.shape) < prob) & ~special
    labels[~selected] = -100                       # score only selected positions

    corrupted = ids.clone()
    r = torch.rand(ids.shape)
    corrupted[selected & (r < 0.8)] = vocab.mask_id
    rand_pos = selected & (r >= 0.8) & (r < 0.9)
    corrupted[rand_pos] = torch.randint(4, len(vocab), (int(rand_pos.sum()),))
    # selected & (r >= 0.9): left unchanged on purpose
    return corrupted, labels


torch.manual_seed(0)
demo = torch.tensor([vocab.encode(pre_texts[0], max_len=MAX_LEN)])
corrupted, labels = mask_tokens(demo, vocab, prob=0.4)   # high prob so we see it
print("original :", vocab.decode(demo[0]))
print("corrupted:", vocab.decode(corrupted[0]))
print("scored at:", [(i, vocab.itos[int(t)])
                     for i, t in enumerate(labels[0]) if t != -100])

# %% [markdown]
# ### Pretrain

# %%
def batch_ids(texts, idxs, max_len):
    return torch.tensor([vocab.encode(texts[i], max_len=max_len) for i in idxs])


cu.set_seed(1337)
enc = Encoder(ecfg).to(device)
opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=0.01)
STEPS, BATCH = 2500, 64                            # unlabelled text only
brng = random.Random(0)

print("pretraining with MLM (no labels used at all):")
for step in range(STEPS + 1):
    idxs = [brng.randrange(len(pre_texts)) for _ in range(BATCH)]
    ids = batch_ids(pre_texts, idxs, MAX_LEN)
    corrupted, labels = mask_tokens(ids, vocab)
    corrupted, labels = corrupted.to(device), labels.to(device)
    _, loss = enc.forward_mlm(corrupted, labels)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
    opt.step()
    if step % 400 == 0:
        print(f"  step {step:5d} | mlm loss {loss.item():.4f} "
              f"(ln(V) = {math.log(len(vocab)):.2f})")

pretrained_state = {k: v.clone() for k, v in enc.state_dict().items()}

# %% [markdown]
# ### Does it fill in blanks?

# %%
enc.eval()
for sentence in ["the play is [MASK] and i shall return",
                 "the king is [MASK] and i left at once",
                 "this book is lovely and [MASK] heart is light"]:
    ids = torch.tensor([vocab.encode(sentence, max_len=MAX_LEN)]).to(device)
    pos = (ids[0] == vocab.mask_id).nonzero()[0, 0]
    with torch.no_grad():
        logits, _ = enc.forward_mlm(ids)
    top = torch.topk(F.softmax(logits[0, pos], dim=-1), 5)
    print(f"{sentence}")
    print("   ", ", ".join(f"{vocab.itos[int(i)]}:{p:.2f}"
                           for p, i in zip(top.values, top.indices)))

# %% [markdown]
# ### The evidence that transfer will work
#
# Before touching a single label: take the four *seen* positive adjectives, average
# their embeddings into a "positive" centroid, do the same for negative, and then
# ask which centroid each **held-out** adjective is closer to.
#
# Nothing in pretraining ever saw a sentiment label. If the held-out adjectives
# land on the right side anyway, MLM has learned polarity purely from the company
# words keep — and a classifier trained on the seen adjectives will transfer.

# %%
with torch.no_grad():
    E = enc.tok_emb.weight.detach()
    emb = lambda w: E[vocab.stoi[w]]
    pos_centroid = torch.stack([emb(w) for w in SEEN_POS]).mean(0)
    neg_centroid = torch.stack([emb(w) for w in SEEN_NEG]).mean(0)

    print(f"{'held-out adjective':<20}{'cos(pos)':>10}{'cos(neg)':>10}{'verdict':>10}")
    correct = 0
    for w, gold in [(w, 1) for w in HELD_POS] + [(w, 0) for w in HELD_NEG]:
        cp = F.cosine_similarity(emb(w), pos_centroid, dim=0).item()
        cn = F.cosine_similarity(emb(w), neg_centroid, dim=0).item()
        pred = int(cp > cn)
        correct += pred == gold
        print(f"{w:<20}{cp:>10.3f}{cn:>10.3f}"
              f"{('pos' if pred else 'neg') + ('' if pred == gold else ' X'):>10}")
    print(f"\n{correct}/{len(HELD_POS) + len(HELD_NEG)} held-out adjectives placed "
          "on the correct side, from unlabelled text alone.")

# %% [markdown]
# ### Bidirectional attention, visualized
#
# Unlike every heatmap so far, this one has **no black triangle** — every token
# attends in both directions. The `[CLS]` column/row is usually distinctive: it
# is where the sentence-level summary accumulates.

# %%
sentence = pre_texts[3]
ids = torch.tensor([vocab.encode(sentence, max_len=MAX_LEN)]).to(device)
with torch.no_grad():
    _, weights = enc.encode(ids, return_weights=True)
toks = [vocab.itos[int(i)] for i in ids[0]]
n_real = sum(t != "[PAD]" for t in toks)
try:
    import matplotlib.pyplot as plt
    W = weights[-1][0].mean(0)[:n_real, :n_real].cpu()
    cu.show_attention(W, tokens=toks[:n_real],
                      title="bidirectional attention (last layer, mean over heads)")
    plt.tight_layout(); plt.show()
except ImportError:
    pass
print("sentence:", sentence)
print("note: attention above the diagonal is allowed here — that is the point.")

# %% [markdown]
# ## 6 · Fine-tuning, four ways
#
# Now the labels come out. We compare:
#
# | strategy | trainable | when to use |
# |---|---|---|
# | **from scratch** | everything, random init | you have plenty of labels |
# | **full fine-tune** | everything, pretrained init | the standard choice |
# | **linear probe** | the classification head only | tiny label sets, or as a diagnostic |
# | **LoRA** | small low-rank adapters | large models, many tasks, little memory |
#
# The interesting question is not "which wins at 5,000 labels" — everything wins
# there. It is **how few labels each strategy needs**, so we sweep the label
# budget instead of fixing it.

# %%
print(f"evaluating every configuration on the same {len(val_lab):,} held-out")
print(f"examples, all of which use adjectives no label ever covered")


def cls_batch(pairs, idxs):
    ids = torch.tensor([vocab.encode(pairs[i][0], max_len=MAX_LEN) for i in idxs])
    y = torch.tensor([pairs[i][1] for i in idxs])
    return ids.to(device), y.to(device)


@torch.no_grad()
def accuracy(model, pairs, pool="cls", n=1500):
    model.eval()
    idxs = list(range(min(n, len(pairs))))
    ids, y = cls_batch(pairs, idxs)
    logits, _ = model.forward_cls(ids, pool=pool)
    return (logits.argmax(-1) == y).float().mean().item()


def finetune(model, train_lab, steps=250, lr=3e-4, trainable=None, pool="cls",
             batch=32, seed=0):
    if trainable is not None:
        for name, p in model.named_parameters():
            p.requires_grad = any(t in name for t in trainable)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    r = random.Random(seed)
    model.train()
    for _ in range(steps):
        idxs = [r.randrange(len(train_lab)) for _ in range(min(batch, len(train_lab)))]
        ids, y = cls_batch(train_lab, idxs)
        _, loss = model.forward_cls(ids, y, pool=pool)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return accuracy(model, val_lab, pool=pool), n_train

# %% [markdown]
# ### Sample efficiency: the whole point of pretraining
#
# `from scratch` and `full fine-tune` are the *same architecture with the same
# number of trainable parameters* — the only difference is whether the weights
# started from MLM pretraining. Any gap is what pretraining bought.

# %%
LABEL_BUDGETS = (8, 16, 32, 64, 200, 400)
sweep = {"from scratch": [], "full fine-tune": []}

for n_labels in LABEL_BUDGETS:
    subset = train_lab[:n_labels]
    cu.set_seed(0)
    scratch = Encoder(ecfg).to(device)
    acc_s, _ = finetune(scratch, subset)

    cu.set_seed(0)
    full = Encoder(ecfg).to(device)
    full.load_state_dict(pretrained_state)
    acc_f, _ = finetune(full, subset)

    sweep["from scratch"].append(acc_s)
    sweep["full fine-tune"].append(acc_f)
    print(f"{n_labels:>4} labels | from scratch {acc_s:6.1%} | "
          f"pretrained {acc_f:6.1%} | gap {acc_f - acc_s:+6.1%}")

# %%
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3.6))
    for name, accs in sweep.items():
        plt.plot(LABEL_BUDGETS, [100 * a for a in accs], "o-", label=name)
    plt.xscale("log"); plt.xticks(LABEL_BUDGETS, [str(b) for b in LABEL_BUDGETS])
    plt.axhline(50, color="gray", ls=":", label="chance")
    plt.xlabel("labelled examples"); plt.ylabel("val accuracy %")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.title("what MLM pretraining buys: sample efficiency")
    plt.tight_layout(); plt.show()
except ImportError:
    pass

print("""
Read that carefully: the from-scratch model does not improve with MORE labels.
It cannot. Its embeddings for the held-out adjectives are still random -- no
label ever mentioned them -- so it is guessing on every validation example,
whether it saw 8 labels or 400.

The pretrained model is at ceiling from 8 labels onward. All it had to learn was
"read the polarity direction that already exists in the embedding space", and a
handful of examples is enough to find it.

This is the sharpest possible version of the real phenomenon. On actual text the
gap is a matter of degree rather than all-or-nothing, because a from-scratch
model can partly generalize through shared subwords and context. But the
mechanism is identical: pretraining supplies representations, and labels only
have to supply the decision boundary.
""")

# %% [markdown]
# ### Linear probe: how good are the frozen features?
#
# A linear probe freezes the entire encoder and trains only the classification
# head — 258 parameters. Whatever accuracy it reaches is accuracy that was
# *already present* in the pretrained representation; the probe only reads it out.
#
# Two ways to summarize a sequence into one vector, worth comparing:
#
# - **`[CLS]` pooling** — use position 0's final hidden state. Note that MLM gives
#   `[CLS]` no direct training signal (we exclude it from masking), so it works
#   only because attention mixes sentence content into it. Real BERT added a
#   second objective, next-sentence prediction, largely to train this position
#   deliberately.
# - **mean pooling** — average over non-padding tokens. Needs no special token and
#   is the usual choice for sentence embeddings.

# %%
FT = train_lab[:64]
probe_results = {}
for pool in ("cls", "mean"):
    cu.set_seed(0)
    probe = Encoder(ecfg).to(device)
    probe.load_state_dict(pretrained_state)
    acc, n_train = finetune(probe, FT, trainable=["cls_head"], lr=1e-2, pool=pool)
    probe_results[pool] = acc
    print(f"  frozen encoder, {pool + '-pooled':<12} probe "
          f"({n_train:,} trainable) -> {acc:.1%}")
print("\nA 258-parameter linear model reaching ~100% means MLM did essentially")
print("all the work. That is the whole premise of the pretrain-then-adapt era.")

# %% [markdown]
# ## 7 · LoRA from scratch
#
# Low-Rank Adaptation freezes `W` and learns a low-rank update:
#
# ```
# h = W x + (B A) x * (alpha / r)      A: (r, in)  B: (out, r)
# ```
#
# With `r = 4` on a `128 x 128` layer, that is 1024 trainable parameters instead
# of 16384. `A` is randomly initialized and `B` is **zeros**, so the adapter
# starts as an exact no-op — fine-tuning begins from the pretrained function.
#
# At the end you can *merge* `W + BA` into a single matrix, so inference costs
# nothing extra. That combination — tiny checkpoints, zero inference overhead —
# is why LoRA is the default way to fine-tune large models.

# %%
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                 # freeze the pretrained weight
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))   # zeros!
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        delta = self.drop(x) @ self.lora_A.T @ self.lora_B.T
        return out + delta * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the adapter into the base weight — zero inference overhead."""
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None)
        merged.weight.copy_(self.base.weight + self.scaling * (self.lora_B @ self.lora_A))
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


base = nn.Linear(128, 128)
lora = LoRALinear(base, r=4)
x = torch.randn(2, 5, 128)
print("starts as an exact no-op:", torch.allclose(lora(x), base(x)))
print(f"base params {cu.count_params(base, False):,} | "
      f"lora trainable {cu.count_params(lora):,} "
      f"({100 * cu.count_params(lora) / cu.count_params(base, False):.1f}%)")

with torch.no_grad():
    lora.lora_B.normal_(0, 0.02)
print("after training, merged == unmerged:",
      torch.allclose(lora(x), lora.merge()(x), atol=1e-5))

# %%
def apply_lora(model, r=4, alpha=8, targets=("qkv", "proj")):
    """Wrap every matching nn.Linear in the model with a LoRA adapter."""
    for name, module in model.named_modules():
        for cname, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in cname for t in targets):
                setattr(module, cname, LoRALinear(child, r, alpha))
    return model


# %% [markdown]
# ### The four strategies at 64 labels

# %%
total_params = cu.count_params(Encoder(ecfg), trainable_only=False)
table = []

cu.set_seed(0)
m = Encoder(ecfg).to(device)
table.append(("from scratch", *finetune(m, FT)))

cu.set_seed(0)
m = Encoder(ecfg).to(device); m.load_state_dict(pretrained_state)
table.append(("full fine-tune", *finetune(m, FT)))

cu.set_seed(0)
m = Encoder(ecfg).to(device); m.load_state_dict(pretrained_state)
table.append(("linear probe (mean)", *finetune(m, FT, trainable=["cls_head"],
                                               lr=1e-2, pool="mean")))

cu.set_seed(0)
lora_model = Encoder(ecfg).to(device)
lora_model.load_state_dict(pretrained_state)
apply_lora(lora_model)
lora_model.to(device)
table.append(("LoRA (r=4)", *finetune(lora_model, FT,
                                      trainable=["lora_", "cls_head"], lr=3e-3)))

print(f"{'strategy':<22}{'val acc':>9}{'trainable':>12}{'share':>9}")
for name, acc, n_train in table:
    print(f"{name:<22}{acc:>8.1%}{n_train:>12,}{n_train / total_params:>8.2%}")
print("""
LoRA should land close to full fine-tuning while training ~1-2% of the weights.
At 7B+ parameters that difference is what makes fine-tuning possible on one GPU
at all -- and the adapter checkpoint is megabytes instead of gigabytes.
""")

# %% [markdown]
# ### Where the classifier looks

# %%
lora_model.eval()
for s, y in [("the play is not vile indeed", 1), ("this book is grim at length", 0)]:
    ids = torch.tensor([vocab.encode(s, max_len=MAX_LEN)]).to(device)
    with torch.no_grad():
        logits, _ = lora_model.forward_cls(ids)
        probs = F.softmax(logits[0], dim=-1)
    print(f"{s!r:<38} -> p(pos) {probs[1]:.3f} (gold {'pos' if y else 'neg'})")

# %% [markdown]
# ## 8 · Compare against a real pretrained model (optional)
#
# Our encoder saw 5,000 template sentences. A real one saw billions of words of
# text. If `transformers` is installed, this is worth running once just to see
# how much the pretraining corpus matters.

# %%
try:
    from transformers import AutoModel, AutoTokenizer

    name = "prajjwal1/bert-tiny"          # 4M params, downloads in seconds
    hf_tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModel.from_pretrained(name)
    print(f"{name}: {cu.human(sum(p.numel() for p in hf.parameters()))} params, "
          f"vocab {hf_tok.vocab_size:,}")
    enc_out = hf(**hf_tok("the play is lovely indeed", return_tensors="pt"))
    print("last_hidden_state:", tuple(enc_out.last_hidden_state.shape))
    print("Same architecture you just built — embeddings, bidirectional blocks,")
    print("[CLS] at position 0 — only bigger and much better pretrained.")
except ImportError:
    print("transformers not installed. `pip install transformers` to try this.")
except Exception as exc:
    print(f"skipped ({type(exc).__name__}): needs network access")

# %% [markdown]
# ## 9 · Choosing an architecture, decided
#
# | you want to... | use | why |
# |---|---|---|
# | generate text | decoder-only | only it factorizes left to right |
# | classify / score / embed | encoder-only | bidirectional context, cheap |
# | map sequence A to B (speech, translation) | encoder-decoder | clean separation |
# | do all of the above with one model | decoder-only + prompting | what the industry chose |
#
# And for adaptation: full fine-tune when the model is small or the task is far
# from pretraining; LoRA when the model is large or you need many task-specific
# variants; a linear probe when you have very few labels or want a fast baseline.

# %% [markdown]
# ## Exercises
#
# 1. **Masking rate.** Sweep MLM `prob` = 0.05, 0.15, 0.40 and compare downstream
#    accuracy. Why does too much masking hurt as much as too little?
# 2. **Drop the 80/10/10.** Use `[MASK]` 100% of the time. Measure the
#    train/fine-tune mismatch this creates.
# 3. **Pooling.** Replace `[CLS]` with mean-pooling over non-pad tokens. Which
#    works better here, and does it change with more pretraining?
# 4. **LoRA rank.** Sweep `r` = 1, 2, 4, 16, 64. Plot accuracy against trainable
#    parameters. Where does it saturate?
# 5. **Which layers matter.** Apply LoRA only to the first two blocks, then only
#    to the last two. Which half of the network needs adapting for this task?
#
# ---
# **Next:** [12 · Inference: KV cache, decoding strategies and quantization](12_inference_and_decoding.ipynb)
