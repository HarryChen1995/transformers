"""Minimal decoder-only Transformer (GPT-style) in PyTorch.

Trains a character-level language model on a small built-in text corpus
and predicts / generates the next tokens autoregressively.

Usage:
    python transformer.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    block_size = 64      # max context length
    d_model = 128        # embedding dim
    n_heads = 4
    n_layers = 4
    dropout = 0.1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# B = batch size, T = sequence length (<= block_size), C = d_model,
# h = n_heads, d = head_dim = C // h, V = vocab_size

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        # one fused Linear producing Q, K, V together: (C) -> (3C)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        # output projection after heads are concatenated: (C) -> (C)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        # lower-triangular causal mask, shape (block_size, block_size),
        # reshaped to (1, 1, block_size, block_size) so it broadcasts
        # over the (B, h, T, T) attention score tensor
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape

        # qkv(x): (B, T, 3C) -> split last dim into 3x (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, T, h, d) -> (B, h, T, d) so attention is per-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, T, d)

        # (B, h, T, d) @ (B, h, d, T) -> (B, h, T, T) raw attention scores,
        # scaled by sqrt(d) to keep softmax gradients well-behaved
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # mask future positions (j > i) to -inf before softmax
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)          # (B, h, T, T), rows sum to 1
        att = self.dropout(att)

        # (B, h, T, T) @ (B, h, T, d) -> (B, h, T, d) weighted sum of values
        out = att @ v                                   # (B, h, T, d)
        # merge heads back: (B, h, T, d) -> (B, T, h, d) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)                           # (B, T, C)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # position-wise MLP: expand to 4C in the hidden layer, then
        # project back down to C (standard transformer 4x expansion ratio)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),   # (B, T, C)  -> (B, T, 4C)
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),   # (B, T, 4C) -> (B, T, C)
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)   # (B, T, C) -> (B, T, C)


class Block(nn.Module):
    """One transformer decoder block: pre-norm attention + pre-norm MLP,
    each wrapped in a residual connection. Shape in == shape out: (B, T, C)."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        # x: (B, T, C) throughout; residual adds keep the dimension fixed
        x = x + self.attn(self.ln1(x))   # (B, T, C) + (B, T, C)
        x = x + self.ff(self.ln2(x))     # (B, T, C) + (B, T, C)
        return x


class TinyGPT(nn.Module):
    """Full decoder-only transformer (GPT-style) language model.

    vocab_size = V, cfg.d_model = C, cfg.block_size = max T, cfg.n_layers = L.
    """

    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        # token id -> C-dim vector; embedding table is (V, C)
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        # position index -> C-dim vector; embedding table is (block_size, C)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        # stack of L identical decoder blocks, each (B, T, C) -> (B, T, C)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # final projection to vocabulary logits: (C) -> (V)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx: (B, T) integer token ids
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)          # (T,)

        # tok_emb(idx): (B, T, C); pos_emb(pos): (T, C) broadcasts over B
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, C)

        for block in self.blocks:
            x = block(x)                                  # (B, T, C) -> (B, T, C)
        x = self.ln_f(x)                                   # (B, T, C)
        logits = self.head(x)                              # (B, T, V)

        loss = None
        if targets is not None:
            # flatten to (B*T, V) vs (B*T,) for cross_entropy over the vocab dim
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx: (B, T0) starting context; grows by 1 column each iteration
        for _ in range(max_new_tokens):
            # keep only the last block_size tokens as context (model's max T)
            idx_cond = idx[:, -self.cfg.block_size:]        # (B, <=block_size)
            logits, _ = self(idx_cond)                      # (B, T, V)
            logits = logits[:, -1, :] / temperature          # (B, V) last-step logits only
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)             # (B, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")  # zero out all but top_k
            probs = F.softmax(logits, dim=-1)                # (B, V) probability dist.
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1) sampled token id
            idx = torch.cat([idx, next_id], dim=1)            # (B, T+1)
        return idx


# ---------------------------------------------------------------------------
# Data (char-level)
# ---------------------------------------------------------------------------

TEXT = """
the quick brown fox jumps over the lazy dog
a transformer predicts the next token given the previous tokens
attention lets every token look at every other token in the sequence
""".strip()

chars = sorted(set(TEXT))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(TEXT), dtype=torch.long)


def get_batch(batch_size, block_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------

def main():
    device = "cpu"
    if torch.cuda.is_available():
        # prefer the Tesla P100 over any other card in the box
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        device = f"cuda:{next((i for i, n in enumerate(names) if 'P100' in n), 0)}"
    torch.manual_seed(0)

    cfg = Config()
    cfg.block_size = min(cfg.block_size, len(data) - 1)

    model = TinyGPT(vocab_size=len(chars), cfg=cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    batch_size = 16
    steps = 500
    model.train()
    for step in range(steps):
        x, y = get_batch(batch_size, cfg.block_size, device)
        logits, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # --- next-token prediction on a prompt ---
    model.eval()
    prompt = "the quick brown"
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(idx)
    next_token_logits = logits[0, -1]
    next_id = torch.argmax(next_token_logits).item()
    print(f"\nPrompt: {prompt!r}")
    print(f"Predicted next char: {itos[next_id]!r}")

    # --- autoregressive generation ---
    out = model.generate(idx, max_new_tokens=60, temperature=0.8, top_k=5)
    print(f"Generated: {decode(out[0].tolist())!r}")


if __name__ == "__main__":
    main()
