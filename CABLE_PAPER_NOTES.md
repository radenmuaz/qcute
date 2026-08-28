# CABLE paper notes

Source: [arxiv.org/html/2503.08067v3](https://arxiv.org/html/2503.08067v3), "Context-aware Biases
for Length Extrapolation". Repo: [github.com/axiomlab/Cable](https://github.com/axiomlab/Cable) --
`gpt2_jax` is a JAX/Flax port of `Cable/src/model_gpt.py`, restricted to 3 of its `pos_method`
options (`rope`, `learnable`, `base`/NoPE) -- see `gpt2_jax/README.md`. This doc is reference
material pulled from the paper itself, not project status -- see `docs/status_tpu.md` for that.

## What the paper is about

CABLE (Context-Aware BiLinear Encoding, the paper's own proposed method) is a learned,
context-dependent positional bias added to attention scores -- compared against every major
position-encoding family on length extrapolation (train at one sequence length, eval at much
longer ones). **This project only ports the plain baselines (rope/learnable/base-NoPE) from
Cable's reference GPT-2 implementation, not CABLE itself** -- CABLE the proposed method is not
implemented here.

## Baselines compared in the paper

- Learnable APE (Vaswani et al. 2017)
- Sinusoidal APE (Vaswani et al. 2017)
- RoPE (Su et al. 2024) -- **the one this project's `pos_method="rope"` ports**
- ALiBi (Press et al. 2021)
- T5-bias (Raffel et al. 2020)
- Kerple (Chi et al. 2022a)
- Fire (Li et al. 2023)
- DAPE / DAPEv2 (Zheng et al. 2024a/b) -- an augmentation layered on top of another method (see
  Table 2), not a standalone position encoding

## RoPE-specific findings

RoPE (and the plain baselines generally) show "initial improvement followed by a significant
decline at longer lengths" once eval length exceeds the training length -- i.e. RoPE extrapolates
poorly past its training context in this paper's setup. No RoPE *variants* (NTK-scaling, YaRN,
etc.) are tested -- RoPE appears only as one fixed baseline, not ablated internally.

## Model configs (Section 4.2)

| Model | n_layer | n_head | n_embd (d_model) | params |
|---|---|---|---|---|
| tiny | 6 | 8 | 512 | 44M |
| small | 12 | ~~10~~ **12 (likely)** | 768 | 124M |
| medium | 24 | 16 | 1024 | 334M |

**Note on `small`'s head count**: the extraction returned `n_head=10`, but `768/10=76.8` isn't an
integer head_dim -- almost certainly a scrape artifact (table/PDF extraction error), not a real
paper value (attention heads must evenly divide d_model). Standard GPT-2-small is 12 heads/64
head_dim, which is what `gpt2_jax`'s own `small_rope_default.py` already uses and what this project
has been training against all session -- treat the paper's "10" as unverified until checked against
the primary source directly (Cable's own `Cable/src/model_gpt.py` config or the PDF's actual table,
not this HTML scrape).

## Training settings (FineWeb-Edu-10B + WikiText-103 runs)

- Sequence length: 1024
- LR: 0.0006 peak, linear warmup 750 steps, cosine decay to 0.00006
- Vocab size: 50304 (GPT-2 BPE, padded from 50257)
- `total_batch_size` (via grad accum): **524,288 tokens** -- the paper-faithful value this
  project's configs were audited against and fixed to match (see `docs/status_tpu.md`'s
  "paper-faithful total_batch_size" note)
- FineWeb-Edu-10B: ~19k steps (~1 epoch); per-device batch size 64 (tiny) / 32 (small) / 16
  (medium) -- exactly the values `gpt2_jax/README.md`'s "Formula to match Cable's paper baseline"
  section already documents and this project's configs use
- WikiText-103: 9k steps (tiny) / 5k (small) / 3k (medium) -- **not used by this project**, which
  trains on FineWeb-Edu-10B only

BERT config (MLDR retrieval eval, Table 3) is out of scope for this project entirely (no BERT
lineage here) -- noted for completeness only: bert-base-uncased, 512 max seq len training, 14k
steps, batch size 32, Adam lr=1e-4.

## Score tables (verbatim from the paper's HTML)

### Table 1 -- FineWeb-Edu-10B & WikiText-103 perplexity (lower is better)

**FineWeb-Edu-10B, GPT-2 Medium (334M), trained @ 1024, eval @ longer lengths:**

| Seq Len | CABLE | ALiBi | Fire | T5-bias | Kerple | **RoPE** |
|---|---|---|---|---|---|---|
| 512 | 17.00 | 17.30 | 17.60 | 17.79 | 17.22 | 17.39 |
| 1024 | 16.52 | 16.79 | 17.11 | 17.26 | 16.70 | 16.89 |
| 2048 | 15.97 | 16.56 | 19.60 | 38.32 | 16.28 | 38.95 |
| 4096 | 15.34 | 16.67 | 101.98 | 243.69 | 16.78 | 146.72 |
| 8192 | 15.41 | 17.23 | 383.08 | 799.53 | 20.32 | **361.26** |
| 15360 | 15.41 | 17.46 | 835.92 | 1450.83 | 26.13 | 691.90 |

**WikiText-103, GPT-2 Tiny, trained @ 1024:**

| Seq Len | CABLE | ALiBi | Fire | T5-bias | Kerple | **RoPE** |
|---|---|---|---|---|---|---|
| 512 | 23.70 | 24.09 | 24.34 | 25.06 | 23.95 | 23.66 |
| 1024 | 22.32 | 22.74 | 22.90 | 23.60 | 22.56 | 22.26 |
| 2048 | 21.48 | 22.05 | 22.68 | 27.64 | 21.72 | 41.40 |
| 4096 | 20.94 | 21.73 | 29.57 | 73.99 | 21.32 | 114.77 |
| 8192 | 20.65 | 21.58 | 54.89 | 198.64 | 21.33 | **220.56** |
| 15360 | 20.33 | 21.30 | 104.79 | 411.09 | 21.58 | 375.62 |

RoPE tracks CABLE/ALiBi closely up to the 1024 training length, then degrades sharply past it
(e.g. FineWeb medium @ 8192: RoPE 361.26 vs. CABLE 15.41 -- ~23x worse), consistent with RoPE's
known weak length-extrapolation behavior. **This project trains and evals within the 1024 training
context (no length-extrapolation eval performed)**, so this specific failure mode is out of scope
for the current ablation runs, but worth knowing if extrapolation testing is ever added later.

### Table 2 -- DAPEv2 augmentation, GPT-2 Small, trained @ 1024

| Seq Len | CABLE | Kerple | DAPEv2+CABLE | DAPEv2+Kerple |
|---|---|---|---|---|
| 512 | 21.17 | 21.41 | 20.41 | 20.56 |
| 1024 | 20.72 | 21.12 | 19.91 | 20.07 |
| 2048 | 20.23 | 22.58 | 19.30 | 19.51 |
| 4096 | 19.60 | 28.04 | 18.55 | 18.79 |
| 8192 | 19.87 | 39.38 | 18.63 | 18.92 |

(No RoPE column in this table -- RoPE isn't paired with DAPEv2 in the paper.)

### Table 3 -- BERT MLDR retrieval, nDCG@10, trained @ 512 (out of scope, no BERT lineage here)

| Seq Len | CABLE | ALiBi | RoPE | Learnable | Sinusoidal |
|---|---|---|---|---|---|
| 512 | 14.96 | 14.02 | 15.16 | 10.42 | 13.57 |
| 1024 | 15.15 | 12.88 | 14.41 | -- | 12.80 |
| 2048 | 16.77 | 14.30 | 10.26 | -- | 1.03 |
| 4096 | 21.36 | 18.71 | 1.17 | -- | 0.00 |
| 8192 | 24.59 | 22.86 | 0.12 | -- | 0.00 |
| 16384 | 25.10 | 23.44 | 0.12 | -- | 0.00 |

## Relevance to this project's runs

This project's `medium_paper_match_b8`/`small_paper_match` (gpt2_jax baselines) target the paper's
own `medium`/`small` FineWeb-Edu-10B configs (1024 seq len, `total_batch_size=524288`, matching
batch sizes) -- so their converged loss/bpb should be roughly comparable to this table's **1024
row** (RoPE 16.89 PPL medium / -- small not in Table 1) once training finishes, not the longer-eval
rows (this project doesn't do length-extrapolation eval). PPL-to-loss: `loss = ln(PPL)` --
e.g. RoPE@1024/medium's 16.89 PPL ≈ 2.83 nats loss, a rough sanity-check ceiling/floor for
`medium_paper_match_b8`'s eventual converged val loss once it completes its ~19k-step run.
