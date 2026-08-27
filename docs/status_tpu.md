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

- **tpu4** (`35.186.15.67`) — fresh `.venv` + `jax[tpu]` set up, 4 chips confirmed. First
  `dataset_preparation.py` attempt **stalled** (2026-08-27, ~58% through the HF `datasets`
  Arrow-build phase, ~54GB written between raw-parquet `hub/` cache + re-encoded `datasets/`
  cache) — process sat in uninterruptible `D` state with `/proc/<pid>/io`'s `write_bytes` frozen
  across multiple checks, same signature as the earlier `prep_fineweb_edu_bytes.py` disk-hang
  incident (see tpu_setup.md). `kill -9` did not clear it immediately; it eventually cleared
  on its own after ~40+ minutes. `~/.cache/huggingface` wiped afterward, node otherwise idle/clean
  (71GB free) — available as a second node once tpu5 confirms a working approach.
- **tpu5** (`35.186.33.7`) — **active**: fresh `.venv` + `jax[tpu]` set up, 4 chips confirmed.
  Retrying `dataset_preparation.py` here (tmux `cable_data_prep`, log `~/cable_data_prep.log`)
  since tpu4's attempt stalled — watching for the same stall signature before trusting it. Once
  shards land in `~/qcute/data/fineweb-edu-10B/`: smoke-test `gpt2_jax/train_gpt.py`, then launch
  the real run using all 4 chips.
- **tpu6** (`35.186.110.50`) — idle, freshly wiped, no run yet.
- **tpu7** (`35.186.34.230`) — idle, freshly wiped. Its last torch_xla run
  (`bytelm_fineweb`'s `small_d512x8_mlp4_1epoch_ctx1024_flash`, now archived) died silently
  sometime between 2026-08-25 19:22 UTC and the 2026-08-27 check-in — no traceback, no stdout log
  file even present, process and its tmux server both gone. Root cause not investigated (the run
  itself is being superseded by the JAX line, not worth chasing) — noted here in case the same
  silent-death pattern recurs on a live run and needs root-causing then.
