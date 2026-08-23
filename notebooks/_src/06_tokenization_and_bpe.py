# %% [markdown]
# # 06 · Tokenization and byte-pair encoding
#
# **Goal:** replace the character tokenizer with the algorithm real LLMs use.
# We implement byte-pair encoding (BPE) from scratch — training the merges,
# encoding, decoding — then measure what it buys and what it costs.
#
# **You will be able to:**
# 1. explain the sequence-length / vocabulary-size trade-off quantitatively
# 2. implement BPE training and encoding in ~60 lines
# 3. handle arbitrary Unicode with a 256-symbol base alphabet and never see OOV
# 4. add special tokens without corrupting the merge table
# 5. explain famous LLM failures (spelling, arithmetic, `SolidGoldMagikarp`)
#    as tokenization artifacts
#
# Prerequisites: notebooks 01–05.

# %%
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path.cwd()
if not (ROOT / "course_utils.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import course_utils as cu

cu.set_seed(1337)
device = cu.get_device()
text = cu.get_text(verbose=False)
print(f"corpus: {len(text):,} chars")

# %% [markdown]
# ## 1 · The trade-off we are trying to beat
#
# | scheme | vocab | tokens per 1k chars | out-of-vocabulary? |
# |---|---|---|---|
# | characters | ~100 | ~1000 | never |
# | words | 100k+ | ~180 | constantly |
# | **subwords (BPE)** | 30k–100k | ~250 | never |
#
# Sequence length is the thing to minimize, because attention is `O(T²)`.
# Vocabulary size is the thing to keep moderate, because the embedding and
# output head are both `V × C`, and the softmax is over `V`.

# %%
char_tok = cu.CharTokenizer(text)
words = text.split()
print(f"char-level : {len(char_tok.encode(text)):>9,} tokens, vocab {char_tok.vocab_size:>7,}")
print(f"word-level : {len(words):>9,} tokens, vocab {len(set(words)):>7,}")

# The word-level failure mode: anything unseen is unrepresentable.
vocab_words = set(words[:len(words) // 2])
unseen = [w for w in words[len(words) // 2:] if w not in vocab_words]
print(f"\nof the second half's words, {len(unseen):,} are unseen by the first half")
print("examples:", unseen[:8], "-> all would become <UNK>")

# %% [markdown]
# ## 2 · Bytes as the base alphabet
#
# Start from UTF-8 **bytes**, not characters. Every possible text is a sequence
# of bytes in `0..255`, so a 256-symbol base alphabet can represent anything ever
# written — emoji, Chinese, mojibake — with zero out-of-vocabulary cases. BPE
# then learns merges *on top* of that.

# %%
s = "héllo 世界 🚀"
raw = list(s.encode("utf-8"))
print(f"{s!r}")
print(f"  {len(s)} characters -> {len(raw)} bytes: {raw}")
print(f"  decoded back: {bytes(raw).decode('utf-8')!r}")
print("\nNote multi-byte characters: '世' is 3 bytes, '🚀' is 4.")
print("A byte-level model must learn to assemble them — usually it does, easily.")

# %% [markdown]
# ## 3 · The BPE algorithm
#
# Repeat `n_merges` times:
#
# 1. count every adjacent pair in the token sequence
# 2. take the most frequent pair `(a, b)`
# 3. mint a new token id for it, and replace all occurrences
#
# The merge list, in order, *is* the tokenizer. Encoding replays the merges;
# decoding expands them back to bytes.

# %%
def get_pair_counts(ids: list[int]) -> Counter:
    """Count adjacent pairs. (Reuse across merges is the obvious optimization.)"""
    return Counter(zip(ids, ids[1:]))


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# Walk through it on a toy string
demo = "the theme of the theatre"
ids = list(demo.encode("utf-8"))
print(f"start: {len(ids)} tokens")
for step in range(4):
    counts = get_pair_counts(ids)
    pair, freq = counts.most_common(1)[0]
    new_id = 256 + step
    ids = merge_pair(ids, pair, new_id)
    a, b = pair
    show = lambda t: bytes([t]).decode("utf-8", "replace") if t < 256 else f"<{t}>"
    print(f"merge {step}: {show(a)!r} + {show(b)!r} (freq {freq}) -> id {new_id}"
          f" | {len(ids)} tokens left")

# %% [markdown]
# ## 4 · A complete BPE tokenizer
#
# Same code, packaged, with `encode` and `decode`. Two details that matter:
#
# - **Encoding must apply merges in training order.** We repeatedly find the
#   pair present in the input with the lowest merge index. Applying merges in the
#   wrong order gives a different (and inconsistent) tokenization.
# - **Decoding is byte concatenation**, with `errors="replace"`, because a token
#   boundary can fall in the middle of a multi-byte character.

# %%
class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}   # pair -> new id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}

    # -- training ---------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            counts = get_pair_counts(ids)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = 256 + i
            ids = merge_pair(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (i < 5 or i % 100 == 0):
                piece = self.vocab[new_id].decode("utf-8", "replace")
                print(f"  merge {i:4d}: {piece!r:<14} freq {freq:>6,} "
                      f"-> {len(ids):,} tokens")
        return self

    # -- inference --------------------------------------------------------
    def encode_ordinary(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = get_pair_counts(ids)
            # the eligible pair that was learned EARLIEST wins
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge_pair(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        if not (allowed_special and self.special):
            return self.encode_ordinary(text)
        # split on special tokens so they are never merged into neighbours
        import re
        pattern = "(" + "|".join(re.escape(t) for t in self.special) + ")"
        out = []
        for chunk in re.split(pattern, text):
            if chunk in self.special:
                out.append(self.special[chunk])
            elif chunk:
                out.extend(self.encode_ordinary(chunk))
        return out

    def decode(self, ids) -> str:
        inv = {v: k for k, v in self.special.items()}
        parts = []
        for i in ids:
            i = int(i)
            if i in inv:
                parts.append(inv[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    def add_special(self, tokens: list[str]):
        base = 256 + len(self.merges)
        for j, t in enumerate(tokens):
            self.special[t] = base + j
        return self

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special)

    def __repr__(self):
        return (f"BPETokenizer(vocab_size={self.vocab_size}, "
                f"merges={len(self.merges)}, special={len(self.special)})")


# %% [markdown]
# ### Train it
#
# Our naive implementation is `O(n)` per merge over the whole corpus, so we train
# on a slice. Production tokenizers (`tiktoken`, HuggingFace `tokenizers`) do
# this in Rust with incremental pair counts, on hundreds of GB.

# %%
train_slice = text[:120_000]
bpe = BPETokenizer()
t0 = time.time()
bpe.train(train_slice, vocab_size=512, verbose=True)
print(f"\n{bpe} in {time.time() - t0:.1f}s")

# %% [markdown]
# ### What did it learn?
#
# The merges recover linguistic structure — common words, suffixes, and the
# `" word"` pattern (space + word) that dominates real tokenizers — without ever
# being told what a word is.

# %%
pieces = [(new_id, bpe.vocab[new_id].decode("utf-8", "replace"))
          for new_id in sorted(bpe.vocab) if new_id >= 256]
print("first 20 merges:", [p for _, p in pieces[:20]])
print("\nlongest learned tokens:")
for tid, piece in sorted(pieces, key=lambda kv: -len(kv[1]))[:12]:
    print(f"  id {tid:>4}  {piece!r}")

# %% [markdown]
# ### Round-trip and compression

# %%
probe = "The king speaks of the crown, and nothing more."
ids = bpe.encode(probe)
print("text  :", probe)
print("ids   :", ids)
print("pieces:", [bpe.decode([i]) for i in ids])
print("decode:", bpe.decode(ids))
assert bpe.decode(bpe.encode(probe)) == probe, "round trip must be exact"

held_out = text[200_000:210_000]
n_bytes = len(held_out.encode("utf-8"))
n_bpe = len(bpe.encode(held_out))
print(f"\nheld-out slice: {n_bytes:,} bytes -> {n_bpe:,} BPE tokens")
print(f"compression ratio: {n_bytes / n_bpe:.2f}x")
print("(GPT-2's 50k-vocab tokenizer averages ~4x on English)")

# %% [markdown]
# ### Unicode still round-trips exactly

# %%
for s in ["héllo", "世界の言葉", "🚀🚀 to the moon", "def f(x):\n    return x**2"]:
    ok = bpe.decode(bpe.encode(s)) == s
    print(f"  {s!r:<34} {len(bpe.encode(s)):>3} tokens  round-trip={ok}")

# %% [markdown]
# ## 5 · Vocabulary size is a real dial
#
# More merges means shorter sequences (cheaper attention) but a bigger embedding
# table and softmax. Let's measure the curve on our corpus.

# %%
sample = text[:40_000]
print(f"{'vocab':>7} {'tokens':>9} {'ratio':>7} {'emb+head params @C=384':>24}")
for v in (256, 320, 512, 1024):
    t = BPETokenizer().train(sample, vocab_size=v)
    n = len(t.encode(sample))
    ratio = len(sample.encode("utf-8")) / n
    params = 2 * v * 384
    print(f"{v:>7} {n:>9,} {ratio:>7.2f}x {params:>24,}")
print("\nDiminishing returns: each doubling of vocab buys less length reduction")
print("but costs linearly more embedding parameters.")

# %% [markdown]
# ## 6 · Special tokens
#
# Models need tokens that never appear in text: `<|endoftext|>` to separate
# documents, padding, and chat-format markers. They are assigned ids *above* the
# merge range and, critically, are split out before merging so that the literal
# string can never be silently produced by merges.

# %%
bpe.add_special(["<|endoftext|>", "<|pad|>"])
print(bpe)
doc = "First doc.<|endoftext|>Second doc."
ids = bpe.encode(doc)
print("ids:", ids)
print("pieces:", [bpe.decode([i]) for i in ids])
print("<|endoftext|> id:", bpe.special["<|endoftext|>"], "(a single token)")

# %% [markdown]
# ## 7 · Tokenization explains a lot of LLM weirdness
#
# The model never sees characters. It sees token ids. That single fact explains
# several famous failure modes.

# %%
print("A. Spelling / character counting")
for w in ["strawberry", "unbelievable"]:
    ids = bpe.encode(w)
    print(f"   {w!r} -> {[bpe.decode([i]) for i in ids]}")
print("   Counting 'r's requires reasoning about pieces, not letters.\n")

print("B. Leading spaces make different tokens")
for w in ["king", " king"]:
    print(f"   {w!r} -> {bpe.encode(w)}  {[bpe.decode([i]) for i in bpe.encode(w)]}")
print("   'king' and ' king' can have unrelated embeddings.\n")

print("C. Numbers split arbitrarily")
for n in ["7", "42", "1999", "31415"]:
    print(f"   {n!r} -> {[bpe.decode([i]) for i in bpe.encode(n)]}")
print("   Arithmetic on inconsistent digit groupings is genuinely hard.\n")

print("D. Rare tokens are undertrained")
print("   A token appearing 3 times in pretraining has a nearly random")
print("   embedding — the origin of glitch tokens like 'SolidGoldMagikarp'.")

# %% [markdown]
# ### How real tokenizers reduce these problems
#
# GPT-2+ pre-split text with a regex before BPE, so merges never cross word
# boundaries or glue digits together. Modern tokenizers also split every number
# into individual digits. Here is the GPT-2 pattern, applied for illustration:

# %%
import re

GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")
demo = "Hello world! It's 1999, and I've got 42 apples."
print(GPT2_PAT.findall(demo))
print("\nBPE then runs INSIDE each chunk, so 'world!' can never become one token.")

# %% [markdown]
# ### Compare against the real thing (optional)
#
# If `tiktoken` is installed, this contrasts our 512-token tokenizer with GPT-2's
# 50,257-token one. Skipped silently otherwise.

# %%
try:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    for s in ["The king speaks of the crown.",
              "def fibonacci(n): return n if n < 2 else fib(n-1)+fib(n-2)"]:
        mine, theirs = bpe.encode(s), enc.encode(s)
        print(f"{s[:40]!r}...")
        print(f"   ours  (V={bpe.vocab_size:>6}): {len(mine):>3} tokens")
        print(f"   gpt-2 (V={enc.n_vocab:>6}): {len(theirs):>3} tokens "
              f"{[enc.decode([t]) for t in theirs][:10]}")
except ImportError:
    print("tiktoken not installed — `pip install tiktoken` to run this comparison")

# %% [markdown]
# ## 8 · Train TinyGPT on BPE tokens
#
# The payoff: with the same `block_size`, a BPE model sees ~2-4x more *text* per
# window. To compare fairly against the char-level model we must convert loss to
# **bits per character**, since the two models predict different units:
#
# ```
# bits per char = loss_nats / ln(2) * (tokens / characters)
# ```
#
# This is the only honest way to compare tokenizers, and it is a mistake people
# make in benchmarks all the time.

# %%
from tinygpt import GPTConfig, TinyGPT

bpe_ids = bpe.encode_ordinary(text)
tokens_per_char = len(bpe_ids) / len(text)
print(f"{len(text):,} chars -> {len(bpe_ids):,} BPE tokens "
      f"({tokens_per_char:.3f} tokens/char)")

bpe_train, bpe_val = cu.train_val_split(bpe_ids)
splits_bpe = {"train": bpe_train, "val": bpe_val}

cu.set_seed(1337)
cfg_bpe = GPTConfig(vocab_size=bpe.vocab_size, block_size=64, d_model=192,
                    n_heads=6, n_layers=4, dropout=0.1)
model_bpe = TinyGPT(cfg_bpe)
print("params:", cu.human(cu.count_params(model_bpe)))
hist_bpe = cu.train(model_bpe, splits_bpe, steps=600, batch_size=32,
                    block_size=64, lr=3e-4, device=device, eval_every=200)

# %%
char_ids = char_tok.encode(text)
ch_train, ch_val = cu.train_val_split(char_ids)
cu.set_seed(1337)
cfg_char = GPTConfig(vocab_size=char_tok.vocab_size, block_size=64, d_model=192,
                     n_heads=6, n_layers=4, dropout=0.1)
model_char = TinyGPT(cfg_char)
hist_char = cu.train(model_char, {"train": ch_train, "val": ch_val}, steps=600,
                     batch_size=32, block_size=64, lr=3e-4, device=device,
                     eval_every=200)

# %%
import math

bpc_bpe = hist_bpe["val"][-1] / math.log(2) * tokens_per_char
bpc_char = hist_char["val"][-1] / math.log(2)
print(f"{'':<12}{'val loss':>10}{'bits/char':>12}{'chars per 64-token window':>28}")
print(f"{'char-level':<12}{hist_char['val'][-1]:>10.4f}{bpc_char:>12.4f}{64:>28}")
print(f"{'BPE':<12}{hist_bpe['val'][-1]:>10.4f}{bpc_bpe:>12.4f}"
      f"{int(64 / tokens_per_char):>28}")
print("\nThe BPE model's per-token loss looks WORSE (more choices per step) while")
print("its bits/char may be better — and it covers several times more text per")
print("window at identical attention cost. That last part is the real win.")

# %%
prompt = "The king"
idx = torch.tensor([bpe.encode(prompt)], device=device)
out = model_bpe.generate(idx, 60, temperature=0.8, top_k=20)
print("BPE model sample:")
print(bpe.decode(out[0].tolist()))

# %% [markdown]
# ## Exercises
#
# 1. **Speed it up.** Our `train` recounts all pairs every merge. Keep an
#    incremental count and only update pairs adjacent to a merge. Measure the
#    speedup at `vocab_size=1024`.
# 2. **Regex pre-split.** Add GPT-2's pattern to `BPETokenizer.train` so merges
#    stay inside chunks. Does the compression ratio go up or down, and are the
#    learned tokens more sensible?
# 3. **Digit splitting.** Force every digit to be its own token. Then train a
#    small model on generated addition problems and compare accuracy against the
#    unsplit tokenizer.
# 4. **Save/load.** Serialize `merges` and `special` to a JSON file, reload into
#    a fresh tokenizer, and assert identical output on 1,000 random substrings.
# 5. **Vocabulary sweep, properly.** For `vocab_size` in 256/512/1024, train a
#    fixed-parameter model and plot **bits per character** (not per-token loss)
#    against vocab size. Where is the optimum for this corpus?
#
# ---
# **Next:** [07 · Positional encodings: sinusoidal, learned, RoPE and ALiBi](07_positional_encodings.ipynb)
