# TPU training status

Living status doc for anything that trains on a TPU node (`tpu4`/`tpu5`/`tpu6`/`tpu7`, all v4-8,
`us-central2-b` — see [TPU.md](../TPU.md)). Update this in place as runs start/finish/change,
don't append a new dated block each time. Setup/how-to lives in
[tpu_setup.md](tpu_setup.md) (torch/torch_xla install, flash-attention,
multichip-hang investigation, FineWeb-Edu byte-level prep) and
[tpu_direct_ssh.md](tpu_direct_ssh.md) (connection setup) — this doc is state, not how-to.

## 2026-08-27: all 4 nodes reset, new JAX/Cable lineage started

All prior torch/torch_xla sweeps on tpu4/5/6/7 (`qcute.bytelm_tpu`, the FineWeb-Edu
`qcute.bytelm_fineweb`/`qcute.bpelm_fineweb` byte/BPE lines) are stopped and archived —
`run.jsonl` pulled back to local `logs/` for every run that had one, then each node's
`checkpoints/`, `logs/`, old `.venv`/`.venv-nightly*`, and FineWeb-Edu datasets were deleted for a
clean slate (76-80GB free on each node afterward). The fineweb-specific scripts/models/configs were moved to `archive_fineweb_1/` and later deleted
outright (2026-08-27) — that byte/BPE-on-FineWeb-Edu line is not being continued; recover them
from git history (`git log --all -- '*bytelm_fineweb*'`) if ever needed again.

**New direction**: port [Cable](https://github.com/axiomlab/Cable) (the CABLE paper's PyTorch
nanoGPT, `Cable/src/model_gpt.py`) to JAX, restricted to 3 of its `pos_method` options — `rope`,
`learnable` (GPT-2's original absolute position embedding), `base` (NoPE, no positional signal
anywhere) — training on FineWeb-Edu 10B via Cable's own `dataset_preparation.py`, used as-is
(unmodified). New code lives in `gpt2_jax/` (`model_gpt.py`, `data_loader.py`, `train_gpt.py`,
plus a copy of `dataset_preparation.py`) — see that module's own docstrings for the port's exact
scope and what's deliberately not ported (ALiBi/CABLE/FIRE/KERPLE/T5Bias/etc.). Multi-device
training uses `jax.pmap` across all locally-addressable TPU chips (JAX doesn't share torch_xla's
confirmed multichip+nightly hang — see tpu_setup.md — so this is expected to actually use
all 4 chips, unlike the torch_xla line ever managed to).

Env: plain `.venv` via `uv sync` + `uv pip install -U "jax[tpu]"` on top (jax-ai-stack's own pin
is CPU-only; the TPU extra bumps jax/jaxlib and pulls in `libtpu`) — no nightly-build split needed
for this lineage, unlike the torch_xla one.

**Disk-stall postmortem + fix** (2026-08-27): both tpu4's and tpu5's first `dataset_preparation.py`
attempts independently stalled the same way — ~55-60GB written between HF `datasets`' raw-parquet
`hub/` cache + re-encoded `datasets/` cache, process stuck in uninterruptible `D` state,
`/proc/<pid>/io`'s `write_bytes` frozen, `kill -9` not clearing it for 30-40+ minutes. Fixed by
defaulting `HF_HOME`/`HF_DATASETS_CACHE` to `/dev/shm` (tmpfs, ~390GB free RAM on these nodes) —
`gpt2_jax/dataset_preparation.py` now sets this via `os.environ.setdefault(...)` before importing
`datasets` — see CLAUDE.md's "Default any TPU-node data prep/dataloader cache to RAM" rule. Both
nodes' prep completed cleanly after the fix (100 shards each, ~19GB, `~/qcute/data/fineweb-edu-10B/`).

`gpt2_jax/train_gpt.py` smoke-tested successfully on tpu4 with the `tiny` model (all 3 pos_methods,
all 4 chips, real data): loss 10.9 -> ~4.5 over 730 steps, ~700K tokens/s. Along the way, fixed a
`jax`/`flax` version mismatch (`jax[tpu]` bumped jax past what jax-ai-stack's pinned `flax==0.12.0`
needs — `uv pip install -U flax` resolved it) and replaced the now-removed `jax.device_put_replicated`
with a `Mesh`/`NamedSharding`-based `replicate()` helper (both fixes are in `train_gpt.py` now).
Added qcute-style `--config path.py` parsing (`configs/gpt2_jax/*.py`, prints what's
added/updated/unrecognized from the config, snapshots the fully-resolved config into
`logs/<run_name>/resolved_config.py`).

**Current target: Cable's `medium` model** (n_layer=24, n_head=16, n_embd=1024, 353.8M params),
not `tiny`/`small` (their configs were deleted — `git log --all -- 'configs/gpt2_jax/tiny*'
'configs/gpt2_jax/small*'` to recover if needed). Two configs: `configs/gpt2_jax/
medium_rope_default.py` (Cable's own hyperparameters unchanged) and `medium_rope_tuned.py`
(doubled total_batch_size + sqrt-scaled lr).

**`medium` OOMs at Cable's own batch_size=16 default on a v4-8 chip's 30.75GB HBM** (plain
attention, no flash) — these TPU chips have far less memory than the paper's 8x H100 80GB.
batch_size=8 came within 6MB of fitting; batch_size=4 (grad_accum=32, keeping Cable's own
total_batch_size=524288) does fit but is slow: steady-state ~10s/step, ~52K tokens/s.

**Fix: `ModelConfig.use_flash_attention`** (this port's own addition, not in Cable's original)
using JAX's Pallas TPU flash-attention kernel (`jax.experimental.pallas.ops.tpu.flash_attention`)
in place of the plain materialize-the-whole-score-matrix path — confirmed correct across all 4
chips via `pmap` on tpu5 (max abs diff 0.031 vs. the plain-attention reference, bf16-level
variance). With flash-attention on: batch_size=16 (Cable's own original per-device default) still
OOMs (identical shortfall, ~160MB — most of the memory pressure is elsewhere, not the attention
score matrix); **batch_size=12 is the largest that fits** and is genuinely faster:
steady-state ~797ms/step, ~61.7K tokens/s (grad_accum=1, total_batch_size=49152 — smaller than
Cable's own 524288 since grad_accum=1 was prioritized over matching that exact total per user
instruction). At that throughput, 1 epoch (10B tokens // 49152 = 203,450 steps) is
**~45 hours (~1.9 days)**.

**Real launch (2026-08-27, tpu4, tmux `medium_flash_c0`)**: `python3 gpt2_jax/train_gpt.py --model
medium --pos-method rope --dataset-dir data/fineweb-edu-10B --use-flash-attention --batch-size 12
--total-batch-size 49152 --run-name medium_rope_flash_b12`, log `~/medium_flash_c0.log` on the
node (`logs/medium_rope_flash_b12/` for the structured JSONL) — target: `gpt2_jax/
CABLE_PAPER_NOTES.md`'s RoPE/medium perplexity ≈16.9 at context=1024.

- **tpu6** (`35.186.110.50`) — env fully set up (`.venv` + `jax[tpu]` + `flax`, 4 chips confirmed),
  otherwise idle, ready for a run.
- **tpu7** (`35.186.34.230`) — env fully set up (`.venv` + `jax[tpu]` + `flax`, 4 chips confirmed),
  otherwise idle. Its last torch_xla run (`bytelm_fineweb`'s
  `small_d512x8_mlp4_1epoch_ctx1024_flash`, now archived) died silently sometime between
  2026-08-25 19:22 UTC and the 2026-08-27 check-in — no traceback, no stdout log file even
  present, process and its tmux server both gone. Root cause not investigated (the run itself is
  superseded by the JAX line, not worth chasing) — noted here in case the same silent-death
  pattern recurs on a live run and needs root-causing then.
