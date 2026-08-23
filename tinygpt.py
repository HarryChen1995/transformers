"""The TinyGPT model built in notebook 05, as an importable module.

Later notebooks (06, 09, 12, 13) and `scripts/train_gpt.py` import from here so
the architecture is defined exactly once. If you change something, change it
here and the notebooks that import it follow.

    from tinygpt import GPTConfig, TinyGPT
    model = TinyGPT(GPTConfig(vocab_size=47))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 47
    block_size: int = 128       # max T
    d_model: int = 192          # C
    n_heads: int = 6            # h
    n_layers: int = 6           # L
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True
    flash: bool = True          # use the fused F.scaled_dot_product_attention

    @property
    def head_dim(self) -> int:  # d
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.dropout_p = cfg.dropout
        self.flash = cfg.flash
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                 .view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x, return_weights: bool = False):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, h, d).transpose(1, 2)               # (B, h, T, d)
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)

        if self.flash and not return_weights:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0)
            att = None
        else:
            att = (q @ k.transpose(-2, -1)) * d ** -0.5      # (B, h, T, T)
            att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = self.attn_drop(att) @ v                      # (B, h, T, d)

        y = y.transpose(1, 2).contiguous().view(B, T, C)     # (B, T, C)
        y = self.resid_drop(self.proj(y))
        return (y, att) if return_weights else y


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
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
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2: shrink the projections that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, return_weights=False):
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))     # (B, T, C)

        attentions = []
        for block in self.blocks:
            if return_weights:
                y, att = block.attn(block.ln1(x), return_weights=True)
                x = x + y
                x = x + block.ff(block.ln2(x))
                attentions.append(att)
            else:
                x = block(x)
        logits = self.head(self.ln_f(x))                          # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return (logits, loss, attentions) if return_weights else (logits, loss)

    def optim_groups(self, weight_decay: float = 0.1):
        """Decay matrices, don't decay biases/norms (notebook 09 explains why)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[..., -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None:
                srt, idx_srt = torch.sort(logits, descending=True, dim=-1)
                p = F.softmax(srt, dim=-1)
                remove = p.cumsum(-1) - p > top_p
                srt = srt.masked_fill(remove, float("-inf"))
                logits = torch.empty_like(logits).scatter_(-1, idx_srt, srt)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        self.train(was_training)
        return idx
