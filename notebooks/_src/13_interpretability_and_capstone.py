# %% [markdown]
# # 13 · Interpretability and the capstone
#
# **Goal:** open the model up and look inside — then put the whole course
# together into a project of your own.
#
# Part A is mechanistic interpretability: what individual heads compute, how the
# residual stream evolves, what happens when you delete a component, and the
# single most famous circuit in the field (induction heads).
#
# Part B is a capstone: a checklist, three project tracks, and where to read next.
#
# **You will be able to:**
# 1. classify attention heads by their behavior, quantitatively
# 2. detect an induction head with a controlled experiment
# 3. use the logit lens to watch a prediction form layer by layer
# 4. ablate heads and layers to measure what the model actually relies on
# 5. plan and execute a transformer project end to end
#
# Prerequisites: notebooks 01–05 at minimum; 06–12 make it richer.

# %%
import math
import sys
import time
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
# ## 0 · A model to dissect
#
# `flash=False` so the attention weights are materialized and inspectable — the
# trade-off from notebook 08.

# %%
cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=128, d_model=192,
                n_heads=6, n_layers=6, dropout=0.1, flash=False)
cu.set_seed(1337)
model = TinyGPT(cfg).to(device)

ck = cu.CKPT_DIR / "tinygpt.pt"
if ck.exists() and cu.load_checkpoint("tinygpt")["config"].get("d_model") == cfg.d_model:
    model.load_state_dict(cu.load_checkpoint("tinygpt")["state_dict"], strict=False)
    print("loaded notebook 05's checkpoint")
else:
    print("training a fresh model (~1 min)")
    cu.train(model, splits, steps=1200, batch_size=32, block_size=cfg.block_size,
             lr=3e-4, device=device, eval_every=600)
model.eval()
baseline = cu.estimate_loss(model, splits, 16, cfg.block_size, device, iters=30)
print("baseline val loss:", round(baseline["val"], 4))

# %% [markdown]
# # Part A · Looking inside
#
# ## 1 · A head taxonomy, measured not guessed
#
# For every head we compute four statistics on real text:
#
# - **prev-token mass** — weight on position `i-1`. High means a "previous token"
#   head, the workhorse of local composition.
# - **self mass** — weight on `i` itself. High often means "do nothing"
#   (attention as a no-op is a real and useful behavior).
# - **mean distance** — how far back it looks on average.
# - **entropy** — low is focused, high is diffuse averaging.

# %%
probe_text = text[1000:1000 + 96]
idx = torch.tensor([tok.encode(probe_text)], device=device)
with torch.no_grad():
    _, _, attentions = model(idx, return_weights=True)
T = idx.size(1)
print("attention tensors:", len(attentions), "layers x", tuple(attentions[0].shape))

dist_mat = (torch.arange(T).view(-1, 1) - torch.arange(T).view(1, -1)).float()
rows = []
for L, A in enumerate(attentions):
    for h in range(A.size(1)):
        w = A[0, h].cpu()
        prev = torch.diagonal(w, offset=-1).mean().item()
        self_m = torch.diagonal(w, offset=0).mean().item()
        mean_d = ((w * dist_mat).sum() / w.sum()).item()
        ent = -(w * w.clamp_min(1e-9).log()).sum(-1).mean().item()
        rows.append((L, h, prev, self_m, mean_d, ent))

print(f"\n{'layer':>5}{'head':>5}{'prev-tok':>10}{'self':>8}{'mean dist':>11}"
      f"{'entropy':>9}  guess")
for L, h, prev, self_m, mean_d, ent in rows:
    if prev > 0.3:
        guess = "previous-token head"
    elif self_m > 0.5:
        guess = "near no-op / self"
    elif ent > 2.5:
        guess = "diffuse averaging"
    elif mean_d > 10:
        guess = "long-range"
    else:
        guess = "local, mixed"
    print(f"{L:>5}{h:>5}{prev:>10.3f}{self_m:>8.3f}{mean_d:>11.2f}{ent:>9.2f}  {guess}")

# %%
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (L, h, *_) in zip(axes.flat, rows[:6]):
        cu.show_attention(attentions[L][0, h][:40, :40].cpu(), ax=ax,
                          title=f"L{L} H{h}")
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %% [markdown]
# ## 2 · Induction heads
#
# The most important circuit discovered so far. An induction head implements
# **"this happened before; what followed it then?"** — it completes the pattern
# `[A][B] ... [A] -> [B]`.
#
# The test is elegant: feed a *random* sequence repeated twice. There is no
# learnable content, so the only way to predict the second copy is to look back
# to the first. Loss on the second half dropping is direct evidence of induction.

# %%
def induction_test(model, seq_len=40, n_trials=20, seed=0):
    """Random sequence, repeated twice. Measures per-position loss."""
    g = torch.Generator().manual_seed(seed)
    losses = torch.zeros(2 * seq_len - 1)
    for _ in range(n_trials):
        half = torch.randint(0, tok.vocab_size, (1, seq_len), generator=g)
        seq = torch.cat([half, half], dim=1).to(device)
        with torch.no_grad():
            logits, _ = model(seq[:, :-1])
        lp = F.cross_entropy(logits[0], seq[0, 1:], reduction="none").cpu()
        losses += lp
    return losses / n_trials


loss_by_pos = induction_test(model)
first = loss_by_pos[:39].mean().item()
second = loss_by_pos[41:].mean().item()
print(f"mean loss, first copy  (nothing to copy from): {first:.4f}")
print(f"mean loss, second copy (repeat visible)      : {second:.4f}")
print(f"ln(V) = {math.log(tok.vocab_size):.4f} (random guessing)")
print(f"\ninduction score (drop from first to second copy): {first - second:+.4f}")
if first - second > 0.3:
    print("=> this model DOES perform in-context copying.")
else:
    print("=> little or no in-context copying in this model. That is an expected")
    print("   negative result at this scale, and worth understanding:")
    print("   * induction requires TWO composed attention layers (a previous-token")
    print("     head, then a head that attends through it). Small, briefly-trained")
    print("     models often have not formed the circuit.")
    print("   * uniformly random token sequences are far off-distribution for a")
    print("     model trained on text; strong learned priors can beat copying.")
    print("   * char-level models must copy character by character, which is a")
    print("     longer-range dependency than a subword model faces.")
    print("   Exercise 2 walks you up the layer count until it appears.")

# %%
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 3.4))
    plt.plot(loss_by_pos, lw=1.2)
    plt.axvline(39, color="red", ls="--", label="second copy starts")
    plt.axhline(math.log(tok.vocab_size), color="gray", ls=":", label="ln(V)")
    plt.xlabel("position"); plt.ylabel("loss"); plt.legend(fontsize=8)
    plt.title("induction test: loss per position on a repeated random sequence")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
except ImportError:
    pass

# %% [markdown]
# ### Which heads do the copying?
#
# On a repeated sequence, an induction head shows a bright diagonal offset by the
# repeat length: position `t` in the second copy attends to position
# `t - seq_len + 1` in the first.

# %%
g = torch.Generator().manual_seed(0)
seq_len = 30
half = torch.randint(0, tok.vocab_size, (1, seq_len), generator=g)
seq = torch.cat([half, half], dim=1).to(device)
with torch.no_grad():
    _, _, atts = model(seq, return_weights=True)

print("attention weight on the 'induction target' (offset by the repeat length):")
scores = []
for L, A in enumerate(atts):
    for h in range(A.size(1)):
        w = A[0, h]
        vals = [w[t, t - seq_len + 1].item() for t in range(seq_len, 2 * seq_len)]
        scores.append((sum(vals) / len(vals), L, h))
for s, L, h in sorted(scores, reverse=True)[:5]:
    print(f"  L{L} H{h}: {s:.3f}")
print("(compare with 1/T ~", f"{1 / (2 * seq_len):.3f}", "for uniform attention)")

# %% [markdown]
# ## 3 · The logit lens
#
# The residual stream is the same shape at every layer, so we can apply the final
# norm and output head to an *intermediate* layer and ask "what would the model
# predict if it stopped here?" It shows predictions forming gradually — early
# layers guess frequent tokens, later layers commit.

# %%
@torch.no_grad()
def logit_lens(model, idx):
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device)
    x = model.tok_emb(idx) + model.pos_emb(pos)
    stages = [("embed", x.clone())]
    for i, block in enumerate(model.blocks):
        x = block(x)
        stages.append((f"block {i}", x.clone()))
    out = []
    for name, h in stages:
        logits = model.head(model.ln_f(h))
        out.append((name, logits))
    return out


prompt = "The king speaks of the cr"
idx = torch.tensor([tok.encode(prompt)], device=device)
stages = logit_lens(model, idx)
print(f"prompt {prompt!r}\n")
print(f"{'stage':<10}{'top-3 next-char predictions':<34}{'entropy':>8}")
for name, logits in stages:
    p = F.softmax(logits[0, -1], dim=-1)
    top = torch.topk(p, 3)
    pretty = ", ".join(f"{tok.itos[int(i)]!r}:{v:.2f}"
                       for v, i in zip(top.values, top.indices))
    ent = -(p * p.clamp_min(1e-9).log()).sum().item()
    print(f"{name:<10}{pretty:<34}{ent:>8.3f}")

# %%
try:
    import matplotlib.pyplot as plt
    final = F.softmax(stages[-1][1][0, -1], dim=-1)
    target = int(final.argmax())
    probs = [F.softmax(l[0, -1], dim=-1)[target].item() for _, l in stages]
    ents = [-(F.softmax(l[0, -1], -1) *
              F.softmax(l[0, -1], -1).clamp_min(1e-9).log()).sum().item()
            for _, l in stages]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(probs, "o-", label=f"p({tok.itos[target]!r})")
    ax.set_xticks(range(len(stages)), [n for n, _ in stages], rotation=45, fontsize=8)
    ax.set_ylabel("probability"); ax.grid(alpha=0.3)
    ax2 = ax.twinx(); ax2.plot(ents, "s--", color="orange", label="entropy")
    ax2.set_ylabel("entropy (nats)")
    ax.set_title("logit lens: a prediction forming through depth")
    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.88), fontsize=8)
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %% [markdown]
# ## 4 · Ablation: what is actually load-bearing?
#
# Delete a component and measure the loss increase. Cheap, causal, and it often
# contradicts what the attention maps suggest — a striking pattern can be
# redundant, and a boring one can be critical.

# %%
@torch.no_grad()
def ablate_head(model, layer: int, head: int, splits, iters=12):
    """Zero one head's output by zeroing its slice of the output projection."""
    attn = model.blocks[layer].attn
    d = model.cfg.head_dim
    W = attn.proj.weight
    saved = W[:, head * d:(head + 1) * d].clone()
    W[:, head * d:(head + 1) * d] = 0
    loss = cu.estimate_loss(model, splits, 16, model.cfg.block_size, device,
                            iters=iters)["val"]
    W[:, head * d:(head + 1) * d] = saved              # restore
    return loss


base = cu.estimate_loss(model, splits, 16, cfg.block_size, device, iters=12)["val"]
print(f"baseline val loss: {base:.4f}\n")
print(f"{'layer':>5}{'head':>5}{'ablated loss':>14}{'damage':>10}")
damage = []
for L in range(cfg.n_layers):
    for h in range(cfg.n_heads):
        l = ablate_head(model, L, h, splits)
        damage.append((l - base, L, h))
        print(f"{L:>5}{h:>5}{l:>14.4f}{l - base:>+10.4f}")

print("""
Note the scale: individual heads barely matter. Within a layer the heads are
largely REDUNDANT -- remove one and the others cover for it, because they were
trained together and the residual stream can route around a missing contribution.
Some ablations even IMPROVE validation loss slightly, which is noise plus mild
regularization, not a discovery. Single-component ablation systematically
understates importance; that is why the layer-level table below looks so
different.""")
print("most load-bearing heads:")
for d_, L, h in sorted(damage, reverse=True)[:3]:
    print(f"  L{L} H{h}: +{d_:.4f} loss when removed")
print("least load-bearing (candidates for pruning):")
for d_, L, h in sorted(damage)[:3]:
    print(f"  L{L} H{h}: {d_:+.4f}")

# %%
@torch.no_grad()
def ablate_layer(model, layer: int, kind: str, splits, iters=12):
    """Skip a whole sublayer by zeroing its output projection."""
    block = model.blocks[layer]
    W = (block.attn.proj if kind == "attn" else block.ff.proj).weight
    saved = W.clone()
    W.zero_()
    loss = cu.estimate_loss(model, splits, 16, model.cfg.block_size, device,
                            iters=iters)["val"]
    W.copy_(saved)
    return loss


print(f"\n{'layer':>5}{'attn removed':>14}{'mlp removed':>14}")
for L in range(cfg.n_layers):
    a = ablate_layer(model, L, "attn", splits)
    m = ablate_layer(model, L, "mlp", splits)
    print(f"{L:>5}{a - base:>+14.4f}{m - base:>+14.4f}")
print("""
Now the structure is unmistakable, and it is the same shape reported for much
larger models:
  * layer 0 attention is irreplaceable -- until it runs, no information has moved
    between positions at all, and every later layer depends on that.
  * middle layers are nearly free to delete. The residual stream simply routes
    around them. This is the empirical basis for layer pruning and early exit.
  * the LAST layers' MLPs matter more than their attention: by then the work is
    turning an assembled representation into a token prediction, which is
    per-position computation, not communication.""")

# %% [markdown]
# ## 5 · The residual stream through depth
#
# Norm growth tells you how much each block writes into the stream. Steady growth
# is healthy; a sudden jump usually means a badly scaled component.

# %%
xb, _ = cu.get_batch(val_data, 8, cfg.block_size, device)
with torch.no_grad():
    pos = torch.arange(cfg.block_size, device=device)
    x = model.tok_emb(xb) + model.pos_emb(pos)
    print(f"{'stage':<12}{'norm':>9}{'attn write':>12}{'mlp write':>11}")
    print(f"{'embed':<12}{x.norm(dim=-1).mean():>9.3f}")
    for i, block in enumerate(model.blocks):
        a = block.attn(block.ln1(x))
        x = x + a
        m = block.ff(block.ln2(x))
        x = x + m
        print(f"{'block ' + str(i):<12}{x.norm(dim=-1).mean():>9.3f}"
              f"{a.norm(dim=-1).mean():>12.3f}{m.norm(dim=-1).mean():>11.3f}")

# %% [markdown]
# ## 6 · What individual neurons respond to
#
# Take an MLP hidden unit and find the text that maximizes it. Some neurons turn
# out to be crisply interpretable (fires on newlines, on capital letters, inside
# quotes); most are polysemantic, which is the central difficulty of the field.

# %%
sample = text[:4000]
ids = torch.tensor([tok.encode(sample)], device=device)
acts = []
with torch.no_grad():
    for start in range(0, ids.size(1) - cfg.block_size, cfg.block_size):
        chunk = ids[:, start:start + cfg.block_size]
        pos = torch.arange(chunk.size(1), device=device)
        x = model.tok_emb(chunk) + model.pos_emb(pos)
        for block in model.blocks[:-1]:
            x = block(x)
        last = model.blocks[-1]
        x_ = x + last.attn(last.ln1(x))
        hidden = F.gelu(last.ff.fc(last.ln2(x_)))        # (1, T, 4C)
        acts.append((start, hidden[0].cpu()))

H = torch.cat([a for _, a in acts], dim=0)               # (N, 4C)
print("hidden activations:", tuple(H.shape))
variance = H.var(dim=0)
for neuron in variance.topk(3).indices.tolist():
    col = H[:, neuron]
    top_pos = col.topk(6).indices.tolist()
    ctxs = []
    for p in top_pos:
        ctx = sample[max(0, p - 12):p + 1].replace("\n", "\\n")
        ctxs.append(f"...{ctx}|")
    print(f"\nneuron {neuron} (var {variance[neuron]:.3f}) fires most after:")
    for c in ctxs:
        print("   ", c)

# %% [markdown]
# ## 7 · The honest caveats
#
# Everything above is *suggestive*, not proof. Known traps:
#
# - **Attention weights are not attribution.** A large weight on a value vector
#   that is near zero moves nothing. Weight × value norm is a better proxy;
#   ablation is better still.
# - **Small models are not small versions of big models.** A 800K-parameter
#   char-level model has different circuits than a 70B one.
# - **Superposition.** Models represent more features than they have dimensions,
#   so single neurons are usually polysemantic. Sparse autoencoders are the
#   current approach to unpacking this.
# - **Confirmation bias.** Any 40×40 heatmap has patterns in it. Predict what you
#   expect *before* you plot, then run an ablation.

# %% [markdown]
# # Part B · Capstone
#
# ## 8 · What you have built
#
# | notebook | you can now build |
# |---|---|
# | 01–02 | tensors, autograd, training loops, an LM baseline |
# | 03–05 | attention, multi-head blocks, a complete GPT |
# | 06–07 | BPE tokenization, four positional schemes |
# | 08 | RMSNorm, SwiGLU, GQA, flash attention (a mini LLaMA) |
# | 09 | schedules, AMP, accumulation, compile, throughput analysis |
# | 10–11 | encoder-decoder with cross-attention, BERT-style MLM, LoRA |
# | 12 | KV cache, every decoding strategy, int8 quantization |
# | 13 | head taxonomy, induction heads, logit lens, ablation |
#
# That is the full stack of a modern language model. What's left is scale and
# data engineering — not new mechanisms.

# %%
components = {
    "scaled dot-product attention": "nb 03",
    "multi-head attention": "nb 04",
    "pre-norm residual block": "nb 04",
    "learned positional embeddings": "nb 05",
    "weight tying + GPT-2 init": "nb 05",
    "byte-pair encoding": "nb 06",
    "sinusoidal / RoPE / ALiBi": "nb 07",
    "RMSNorm, SwiGLU, GQA": "nb 08",
    "warmup+cosine, AMP, grad accum": "nb 09",
    "cross-attention, padding masks": "nb 10",
    "masked LM, LoRA": "nb 11",
    "KV cache, top-p, beam search, int8": "nb 12",
}
print("implemented from scratch in this course:")
for k, v in components.items():
    print(f"  [x] {k:<36} {v}")

# %% [markdown]
# ## 9 · Three project tracks
#
# Pick one and take it to a real result. Each is sized for a weekend on a single
# GPU.
#
# ### Track A — Train the best small LM you can
# Fix a budget (say 30 GPU-minutes) and maximize validation bits-per-character on
# a corpus you care about. Deliverables: a table of ablations (tokenizer, depth vs
# width, LR, RoPE vs learned), the loss curves, and a paragraph on what mattered.
#
# ```bash
# python scripts/train_gpt.py --steps 8000 --block_size 256 --d_model 384 \
#     --n_layers 6 --n_heads 6 --batch_size 64 --lr 6e-4 --amp --compile
# ```
#
# ### Track B — Fine-tune for a task
# Take a pretrained checkpoint (yours, or a small HF model), build a labelled
# dataset, and compare full fine-tuning against LoRA at several ranks. Report
# accuracy, trainable parameters, wall-clock and peak memory.
#
# ### Track C — Serve it fast
# Take a trained model and optimize inference: KV cache, batching, int8, a draft
# model for speculative decoding. Report tokens/s and p50/p99 latency at batch
# 1/8/32, and prove output quality is unchanged.

# %% [markdown]
# ## 10 · A checklist you can reuse for any run
#
# Before training:
# - [ ] initial loss ≈ `ln(V)`
# - [ ] can overfit a single batch to ~0
# - [ ] train/val split has no leakage
# - [ ] shapes asserted at every reshape
# - [ ] seed set, config logged
#
# During training:
# - [ ] both losses logged, plus LR and pre-clip grad norm
# - [ ] tokens/s and (on CUDA) MFU logged
# - [ ] checkpoint on best val, and resumable (optimizer state included)
# - [ ] a sample generated at each eval, to read with your own eyes
#
# After training:
# - [ ] compare against a dumb baseline (bigram, or ln(V))
# - [ ] report bits-per-character if you changed tokenizer
# - [ ] measure inference latency the way you will actually serve it
# - [ ] ablate the one component you were least sure about

# %%
def preflight(model, splits, tok, device, steps=200):
    """Run the pre-training checks from section 10 automatically."""
    report = {}
    xb, yb = cu.get_batch(splits["train"], 8, model.cfg.block_size, device)
    with torch.no_grad():
        _, loss0 = model(xb, yb)
    expected = math.log(tok.vocab_size)
    report["initial_loss"] = loss0.item()
    report["ln_V"] = expected
    report["init_ok"] = abs(loss0.item() - expected) < 0.5

    m = type(model)(model.cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(steps):
        _, l = m(xb, yb)
        opt.zero_grad(set_to_none=True); l.backward(); opt.step()
    report["overfit_loss"] = l.item()
    report["overfit_ok"] = l.item() < 0.2
    return report


rep = preflight(TinyGPT(cfg).to(device), splits, tok, device)
for k, v in rep.items():
    mark = "" if not k.endswith("_ok") else ("  PASS" if v else "  FAIL")
    print(f"  {k:<16} {v if not isinstance(v, float) else round(v, 4)}{mark}")

# %% [markdown]
# ## 11 · Where to go next
#
# **Papers, in the order they make sense after this course**
# - *Attention Is All You Need* (2017) — you have now implemented all of it
# - *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)
# - *Scaling Laws for Neural Language Models* (2020) and *Chinchilla* (2022)
# - *LLaMA* / *LLaMA 2* — the architecture from notebook 08, at scale
# - *FlashAttention* — why notebook 08's one-line change matters
# - *LoRA* (2021), *QLoRA* (2023)
# - *In-context Learning and Induction Heads* (2022) — notebook 13's experiment
# - *Training Compute-Optimal…* plus any recent open model's tech report
#
# **Codebases worth reading, in increasing size**
# - `nanoGPT` (Karpathy) — the natural next step after `tinygpt.py`
# - `minGPT`, `llm.c` — same ideas, different angles
# - `gpt-fast` — inference optimization, notebook 12 taken seriously
# - `litgpt`, `torchtitan` — real distributed training
# - HuggingFace `transformers` — read `modeling_llama.py` and recognize everything
#
# **Topics this course deliberately left out**
# - distributed training (DDP, FSDP, tensor/pipeline parallelism)
# - RLHF / DPO and instruction tuning
# - multimodal models (ViT, CLIP, audio)
# - state-space models (Mamba) and other attention alternatives
# - long-context methods beyond RoPE scaling
# - data curation and deduplication, which matters more than architecture

# %%
print("""
Final word.

Every architectural choice in a modern LLM is a small, checkable idea. You have
implemented each one, measured it, and in several cases watched it fail when
implemented wrong. That is the difference between having read about transformers
and being able to build one.

Go train something.
""")

# %% [markdown]
# ## Exercises
#
# 1. **Predict, then ablate.** Pick the head with the most striking attention map
#    and predict its ablation damage before measuring. Were you right?
# 2. **Grow induction.** Train a 1-layer, 2-layer and 4-layer model and run the
#    induction test on each. Confirm that induction needs at least two layers.
# 3. **Weight x value.** Recompute the head taxonomy using
#    `attention_weight * ||value||` instead of the raw weight. Do any
#    conclusions change?
# 4. **Prune.** Remove the four least load-bearing heads permanently, fine-tune
#    briefly, and compare loss and speed against the full model.
# 5. **Capstone.** Do one of the three tracks in section 9 and write a one-page
#    report using the checklist in section 10.
#
# ---
# **Back to:** [the course index](../README.md)
