# Notes from the CABLE paper (target numbers to reproduce)

Source: [Context-aware Biases for Length Extrapolation](https://arxiv.org/html/2503.08067v3)
(axiomlab/Cable, arXiv 2503.08067v3). This repo's `gpt2_jax/` ports that paper's
`Cable/src/model_gpt.py` to JAX, restricted to `rope`/`learnable`/`base` (NoPE) — see
`model_gpt.py`'s own docstring. This file is a reference target, not exhaustive coverage of the
paper (CABLE/ALiBi/FIRE/KERPLE/T5-bias results are here only as context for where RoPE/Learnable
sit relative to them, not something this port implements or needs to match).

**The `Cable/` reference clone (`git clone https://github.com/axiomlab/Cable.git`) used to build
this port is no longer present in this repo's working tree** (was untracked, later removed) — this
file plus `gpt2_jax/`'s own module docstrings are what's left of that reference; re-clone from
GitHub if a source-level check against the original PyTorch is ever needed again.

## Original training setup (paper's own, 8x H100 80GB)

- Dataset: FineWeb-Edu-10B (9.9B train tokens, 0.1B eval)
- Sequence length: 1024 (all extrapolation eval lengths are evaluated on models trained at 1024)
- Steps: ~19,000 (~1 epoch) — matches this port's own `NUM_DATASET_TOKENS // total_batch_size`
- Batch sizes: 64 (Tiny), 32 (Small), 16 (Medium) — per-GPU micro batch; effective batch reaches
  524,288 tokens via grad accumulation — matches this port's `MICRO_BATCH_SIZES` /
  `TOTAL_BATCH_SIZE_DEFAULT` exactly
- LR: 6e-4 peak, linear warmup over 750 steps (this port uses 715, Cable's own `train_gpt.py`
  constant — the paper's prose rounds it), cosine decay to 6e-5 (0.1x peak) — matches this port's
  `MAX_LR`/`MIN_LR`/`WARMUP_STEPS`
- Vocab: 50,304 (padded up from GPT-2's 50,257) — matches `ModelConfig.vocab_size` default here

## Model shapes reported vs. this port's `MODEL_SHAPES`

| size | paper (n_layer, n_head, n_embd, params) | this port | note |
|---|---|---|---|
| Tiny | 6, 8, 512, 44M | 6, 8, 512, 44.67M | matches |
| Small | 12, 10, 768, 124M | 12, **12**, 768, 123.7M | paper says n_head=10, Cable's own `train_gpt.py` source hardcodes n_head=12 for "small" — this port follows the source code, not the paper prose (10 doesn't divide 768 evenly for a standard head_dim=64 split as cleanly as 12 does: 768/12=64, 768/10=76.8) |
| Medium | 24, 16, 1024, 334M | 24, 16, 1024, 353.8M | shape matches; param count differs (334M paper vs. 353.8M measured here) — not yet root-caused, possibly a non-embedding-vs-total-params counting difference; shape (layer/head/embd) is what actually matters for reproduction, and that matches exactly |

## Perplexity to reproduce, context=1024 (this port's own training context — the paper's own
extrapolation-to-longer-context numbers, 2048+, are NOT reproduction targets for this port since
nothing here changes context length after training; included below only for orientation)

Table 4 of the paper, seq_length=1024 row only (the trained length):

| model | CABLE | ALiBi | Fire | T5-bias | Kerple | **RoPE** | **Learnable** | Sinusoidal |
|---|---|---|---|---|---|---|---|---|
| Tiny | 28.73 | 29.25 | 29.56 | 30.08 | 28.95 | **28.81** | **30.11** | 30.03 |
| Small | 20.63 | 20.99 | 21.26 | 21.57 | 20.86 | **20.87** | **21.56** | 21.83 |
| Medium | 16.52 | 16.79 | 17.11 | 17.26 | 16.70 | **16.89** | not reported for Medium in the paper's own table | — |

**This port's `rope`/`learnable` targets**: perplexity ≈ 28.8 / 30.1 (Tiny), ≈ 20.9 / 21.6
(Small), ≈ 16.9 (Medium, RoPE only) at the paper's own eval protocol (context=1024, 1 epoch over
FineWeb-Edu-10B). `base` (NoPE) has no paper number to target — not reported there (the paper's
own related-work section discusses NoPE but doesn't train/eval it on FineWeb-Edu) — so `base`'s
result here is exploratory, not a reproduction.

Perplexity vs. this port's own logged metric: `train_gpt.py` logs cross-entropy loss (nats),
`ppl = math.exp(loss)` — directly comparable to the table above once val loss stabilizes near the
end of a real run (the paper's own numbers are final-checkpoint eval, not training-curve
snapshots).

Full extrapolation tables (seq_length up to 15360, showing RoPE/Learnable/Sinusoidal collapsing
past their trained context while CABLE/ALiBi/Kerple degrade far more gracefully) are in the
paper's Table 4 directly — not reproduced here since out of this port's scope (no context-length
extrapolation eval implemented in `gpt2_jax/` yet).
