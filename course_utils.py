"""Shared helpers for the Transformers-from-scratch course notebooks.

Nothing here is magic: every function is short and the concepts it wraps are
taught explicitly in the notebook where they first appear. The module exists so
that later notebooks do not have to re-type data loading and training loops.

Typical use inside a notebook:

    import course_utils as cu
    text = cu.get_text()
    tok = cu.CharTokenizer(text)
"""

from __future__ import annotations

import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


# ---------------------------------------------------------------------------
# Reproducibility and devices
# ---------------------------------------------------------------------------

def set_seed(seed: int = 1337) -> None:
    """Seed Python, NumPy-free torch CPU and all accelerators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# On a multi-GPU box, train on the card whose name contains this string.
# Override with e.g. COURSE_GPU="RTX 2060"; falls back to cuda:0 if no match.
PREFERRED_GPU = os.environ.get("COURSE_GPU", "P100")


def preferred_cuda_index() -> int:
    """Index of the GPU matching PREFERRED_GPU, else 0."""
    for i in range(torch.cuda.device_count()):
        if PREFERRED_GPU.lower() in torch.cuda.get_device_name(i).lower():
            return i
    return 0


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device: CUDA (the P100) > Apple MPS > CPU."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        # set_device too, so bare "cuda", autocast and synchronize() follow it
        idx = preferred_cuda_index()
        torch.cuda.set_device(idx)
        return torch.device("cuda", idx)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device | str) -> str:
    device = torch.device(device)
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"


def count_params(module: torch.nn.Module, trainable_only: bool = True) -> int:
    ps = module.parameters()
    if trainable_only:
        ps = (p for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in ps)


def human(n: float) -> str:
    """Format a number as 1.2K / 3.4M / 5.6B."""
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= size:
            return f"{n / size:.2f}{unit}"
    return str(n)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def _synthetic_corpus(n_lines: int = 6000, seed: int = 0) -> str:
    """Deterministic play-script-like corpus, used when the download fails.

    It is not Shakespeare, but it has the properties a language model needs to
    show off: a small alphabet, repeated speaker names, punctuation, and
    phrase-level regularities that a model can actually learn.
    """
    rng = random.Random(seed)
    speakers = [
        "FIRST CITIZEN", "SECOND CITIZEN", "KING", "QUEEN", "MENENIUS",
        "MARCIUS", "NURSE", "MESSENGER", "DUKE", "FRIAR",
    ]
    subjects = ["the king", "my lord", "this night", "our city", "the letter",
                "thy brother", "the tempest", "a shadow", "this hand", "the crown"]
    verbs = ["speaks of", "shall bury", "doth wound", "commands", "hath broken",
             "forgives", "remembers", "denies", "carries", "sets free"]
    objects = ["the peace we swore", "no honest man", "a thousand griefs",
               "the gates of Rome", "my weary heart", "such heavy news",
               "the morning star", "an idle promise", "these good citizens",
               "what love remains"]
    tails = [".", ".", ".", "!", "?", ", and nothing more.", ", I fear.",
             ", my friend."]

    lines: list[str] = []
    for i in range(n_lines):
        if i % 7 == 0:
            lines.append("")
            lines.append(rng.choice(speakers) + ":")
        lines.append(
            f"{rng.choice(subjects).capitalize()} {rng.choice(verbs)} "
            f"{rng.choice(objects)}{rng.choice(tails)}"
        )
    return "\n".join(lines).strip() + "\n"


def get_text(path: str | Path | None = None, verbose: bool = True) -> str:
    """Return the course corpus as one big string.

    Order of preference:
      1. `data/input.txt` if it already exists (so you can drop in your own text)
      2. download tiny-shakespeare (~1.1 MB) and cache it there
      3. a deterministic synthetic corpus, if there is no network
    """
    target = Path(path) if path else DATA_DIR / "input.txt"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        text = target.read_text(encoding="utf-8")
        if verbose:
            print(f"loaded {len(text):,} chars from {target}")
        return text

    try:
        with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=15) as r:
            text = r.read().decode("utf-8")
        target.write_text(text, encoding="utf-8")
        if verbose:
            print(f"downloaded tiny-shakespeare -> {target} ({len(text):,} chars)")
    except Exception as exc:  # offline, SSL policy, proxy, ...
        text = _synthetic_corpus()
        target.write_text(text, encoding="utf-8")
        if verbose:
            print(f"download failed ({type(exc).__name__}); "
                  f"wrote synthetic corpus -> {target} ({len(text):,} chars)")
    return text


# ---------------------------------------------------------------------------
# Character-level tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    """The simplest possible tokenizer: one token per unique character."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def train_val_split(ids: list[int] | torch.Tensor, frac: float = 0.9):
    data = torch.as_tensor(ids, dtype=torch.long)
    n = int(frac * len(data))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int,
              device: torch.device | str = "cpu",
              generator: torch.Generator | None = None):
    """Sample `batch_size` random windows: x = tokens, y = tokens shifted by 1."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,),
                       generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, splits: dict[str, torch.Tensor], batch_size: int,
                  block_size: int, device, iters: int = 40) -> dict[str, float]:
    """Average the loss over a few random batches of each split."""
    was_training = model.training
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train(was_training)
    return out


def train(model, splits, *, steps: int = 500, batch_size: int = 32,
          block_size: int = 64, lr: float = 3e-4, device=None,
          eval_every: int = 100, eval_iters: int = 20, grad_clip: float = 1.0,
          weight_decay: float = 0.1, log: bool = True):
    """A compact but honest training loop; returns a history dict for plotting."""
    device = device or get_device()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.95))
    history = {"step": [], "train": [], "val": [], "step_loss": []}
    t0 = time.time()
    model.train()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            ev = estimate_loss(model, splits, batch_size, block_size, device,
                               iters=eval_iters)
            history["step"].append(step)
            history["train"].append(ev.get("train", float("nan")))
            history["val"].append(ev.get("val", float("nan")))
            if log:
                msg = " | ".join(f"{k} {v:.4f}" for k, v in ev.items())
                print(f"step {step:5d} | {msg} | {time.time() - t0:5.1f}s")
        if step == steps:
            break
        x, y = get_batch(splits["train"], batch_size, block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        history["step_loss"].append(loss.item())
    return history


# ---------------------------------------------------------------------------
# Plotting (matplotlib is optional; these degrade to text)
# ---------------------------------------------------------------------------

def plot_history(history, title: str = "training", ax=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; final losses:",
              history["train"][-1], history["val"][-1])
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    if history.get("step_loss"):
        ax.plot(history["step_loss"], alpha=0.25, lw=0.8,
                label="per-step train loss")
    ax.plot(history["step"], history["train"], "o-", label="train (eval)")
    if not all(math.isnan(v) for v in history["val"]):
        ax.plot(history["step"], history["val"], "s-", label="val (eval)")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def show_attention(weights, tokens=None, ax=None, title="attention"):
    """Heatmap of one (T, T) attention matrix. Rows = queries, cols = keys."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(weights)
        return None
    w = torch.as_tensor(weights).detach().float().cpu()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(w, cmap="viridis")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("key position (attended to)")
    ax.set_ylabel("query position")
    if tokens is not None:
        labels = [repr(t)[1:-1] if isinstance(t, str) else str(t) for t in tokens]
        ax.set_xticks(range(len(labels)), labels, fontsize=7, rotation=90)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return ax


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None


def filter_logits(logits: torch.Tensor, temperature: float = 1.0,
                  top_k: int | None = None, top_p: float | None = None):
    """Apply temperature, then top-k, then nucleus (top-p) filtering."""
    logits = logits / max(temperature, 1e-6)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return logits


def save_checkpoint(model, name: str, **extra):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), **extra}, path)
    print("saved", path)
    return path


def load_checkpoint(name: str, map_location="cpu"):
    path = CKPT_DIR / f"{name}.pt"
    return torch.load(path, map_location=map_location, weights_only=False)
