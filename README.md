# Transformers from Scratch — a PyTorch course

A hands-on course that builds a modern language model one component at a time, in
PyTorch, with nothing hidden behind a library call. Thirteen notebooks take you
from `torch.tensor` to a KV-cached, quantized, RoPE-and-GQA model you can
interpret and serve.

Every notebook is **runnable and self-contained**, prints real numbers, and ends
with exercises. Every architectural claim is measured rather than asserted — when
a component helps, you see the loss drop; when it's implemented wrong, you see
what breaks.

```
laptop (CPU / Apple MPS)          GPU box (CUDA)
  notebooks 01-13, all of them      scripts/train_gpt.py for long runs
  minutes per training cell         the same code, more steps
```

---

## Quick start

```bash
pip install -r requirements.txt
jupyter lab                 # then open notebooks/01_pytorch_foundations.ipynb
```

Optional but recommended — drop your own corpus in place of the default:

```bash
mkdir -p data && cp /path/to/your.txt data/input.txt
```

Without it, `course_utils.get_text()` downloads tiny-shakespeare (~1.1 MB) on
first use, and falls back to a deterministic synthetic corpus if there's no
network — so the notebooks always run.

Smoke-test the toolchain in one line:

```bash
python3 scripts/train_gpt.py --steps 100 --eval_every 50 --d_model 128 --n_layers 2
```

---

## The course

Read in order — each notebook builds on the ones before it. Times are for a
laptop; a GPU is 3-10x faster.

| # | Notebook | What you build | Time |
|---|---|---|---|
| 01 | [PyTorch foundations](notebooks/01_pytorch_foundations.ipynb) | tensors, shapes, broadcasting, autograd, `nn.Module`, the training loop | 10 min |
| 02 | [Language modeling and bigrams](notebooks/02_language_modeling_and_bigrams.ipynb) | tokenizing text, next-token batches, a bigram LM, perplexity | 15 min |
| 03 | [Attention from scratch](notebooks/03_attention_from_scratch.ipynb) | causal averaging → Q/K/V → scaled dot-product attention, visualized | 20 min |
| 04 | [Multi-head attention and blocks](notebooks/04_multihead_attention_and_blocks.ipynb) | fused QKV, feed-forward, residuals, pre-norm vs post-norm | 20 min |
| 05 | [Build and train TinyGPT](notebooks/05_build_and_train_tinygpt.ipynb) | **a complete GPT**: positions, weight tying, GPT-2 init, generation, checkpoints | 25 min |
| 06 | [Tokenization and BPE](notebooks/06_tokenization_and_bpe.ipynb) | byte-pair encoding from scratch, special tokens, bits-per-char comparison | 25 min |
| 07 | [Positional encodings](notebooks/07_positional_encodings.ipynb) | sinusoidal, learned, **RoPE**, **ALiBi** + a length-extrapolation experiment | 25 min |
| 08 | [Modern architecture](notebooks/08_modern_architecture.ipynb) | **RMSNorm, SwiGLU, GQA, flash attention** — a miniature LLaMA, plus MoE | 30 min |
| 09 | [Training at scale](notebooks/09_training_at_scale.ipynb) | warmup+cosine, param groups, clipping, accumulation, AMP, `compile`, MFU | 30 min |
| 10 | [Encoder-decoder seq2seq](notebooks/10_encoder_decoder_seq2seq.ipynb) | cross-attention, padding masks, teacher forcing — with a legible alignment map | 30 min |
| 11 | [BERT-style encoders and fine-tuning](notebooks/11_bert_and_finetuning.ipynb) | masked LM pretraining, `[CLS]` classification, **LoRA from scratch** | 30 min |
| 12 | [Inference and decoding](notebooks/12_inference_and_decoding.ipynb) | **KV cache**, top-k/top-p/min-p, beam search, int8 quantization | 30 min |
| 13 | [Interpretability and capstone](notebooks/13_interpretability_and_capstone.ipynb) | head taxonomy, **induction heads**, logit lens, ablation, project tracks | 30 min |

**Short on time?** 01 → 03 → 05 gets you a working GPT you understand.
**Already know the basics?** Start at 06 and work forward.

---

## What's in the repo

```
transformer.py        the original minimal decoder-only GPT (~240 lines, one file)
tinygpt.py            notebook 05's model as an importable module; used by 06, 09, 12, 13
course_utils.py       shared helpers: corpus, tokenizer, batching, train loop, plots
scripts/train_gpt.py  CLI trainer with AMP, torch.compile, accumulation, checkpointing
notebooks/*.ipynb     the course
notebooks/_src/*.py   the notebook sources (percent format) — see "Editing" below
tools/                build_notebooks.py: _src/*.py -> notebooks/*.ipynb
data/, checkpoints/, runs/    generated, git-ignored
```

`transformer.py` still runs standalone (`python3 transformer.py`) and is the
thing notebook 05 makes fully transparent.

---

## Running on a GPU box over ssh

The notebooks pick the best device automatically (`cuda` > `mps` > `cpu`), so
nothing needs changing. For long runs, use the script instead of a notebook:

```bash
# install the CUDA build there (see the note in requirements.txt)
pip install --index-url https://download.pytorch.org/whl/cu124 torch

rsync -av --exclude data --exclude checkpoints ./ gpu-box:~/transformer/
ssh gpu-box
cd ~/transformer

mkdir -p runs
nohup python scripts/train_gpt.py \
    --steps 20000 --block_size 256 --d_model 384 --n_layers 6 --n_heads 6 \
    --batch_size 64 --grad_accum 2 --lr 6e-4 --warmup 500 \
    --amp --compile --eval_every 500 --out gpt_medium \
    > runs/gpt_medium.log 2>&1 &

tail -f runs/gpt_medium.log
```

Pull the results back and inspect them in any notebook:

```bash
rsync -av gpu-box:~/transformer/checkpoints/gpt_medium.pt checkpoints/
rsync -av gpu-box:~/transformer/runs/ runs/
```

```python
import course_utils as cu
from tinygpt import GPTConfig, TinyGPT

ck = cu.load_checkpoint("gpt_medium")
model = TinyGPT(GPTConfig(**{k: v for k, v in ck["config"].items()}))
model.load_state_dict(ck["state_dict"])
```

`python3 scripts/train_gpt.py --help` lists every flag. Notebook 09 explains what
each one does and how to tell whether it's helping.

---

## Reading the numbers

Two things to keep in mind so the results don't mislead you:

**Loss depends on your corpus.** Tiny-shakespeare is hard (a good small model
reaches ~1.5 nats/char); the synthetic fallback is repetitive and much easier
(under 0.5). Never compare a loss across corpora — only across models trained on
the same data, which is what each notebook does.

**Compare like with like across tokenizers.** A BPE model's per-token loss looks
worse than a char model's simply because it has more choices per step. Convert to
**bits per character** before drawing conclusions (notebook 06 does this).

Sanity checks worth internalizing: an untrained model should sit at `ln(V)`, and
any healthy model should be able to overfit a single batch to ~0. Notebook 09
turns both into an automated preflight check.

---

## Editing the notebooks

Notebooks are generated from percent-format Python sources so that every cell can
be executed and checked as a plain script:

```bash
# edit notebooks/_src/05_build_and_train_tinygpt.py, then:
python3 tools/build_notebooks.py 05        # rebuild just that one
python3 tools/build_notebooks.py           # rebuild all

MPLBACKEND=Agg python3 notebooks/_src/05_build_and_train_tinygpt.py   # verify it runs
```

Editing the `.ipynb` directly in Jupyter is fine too — the `_src` file is then
the stale copy, so pick one direction and stick to it.

---

## Requirements

Python 3.10+, PyTorch 2.4+, JupyterLab, matplotlib, numpy — see
`requirements.txt`. Two notebooks have optional extras (`tiktoken` in 06,
`transformers` in 11); both skip themselves cleanly if the package is missing.

## Prerequisites

Comfortable Python. Some memory of what a derivative and a matrix product are.
No prior deep learning experience assumed — notebook 01 starts from
`torch.tensor([1.0])`.
