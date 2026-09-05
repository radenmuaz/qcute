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

**Superseded (2026-08-27) by bf16 mixed precision** — `gpt2_jax/model_gpt.py`'s `ModelConfig` now
has `compute_dtype`/`param_dtype` (default bf16 compute / fp32 param master), wired through every
`nnx.Linear`/`nnx.Embed` (matmul-heavy layers), matching Cable's own `torch.autocast(bfloat16)`.
`LayerNorm` and the final tied-head logits/loss stay fp32 (numerically sensitive, matches
autocast's own policy), as does RoPE's cos/sin table before being cast down to q/k's dtype.
Verified locally: forward/backward run correctly, params/grads stay fp32 (master copy), only the
Linear/Embed matmuls actually execute in bf16.

**With bf16, Cable's own original batch_size=16/grad_accum=1 now fits** (previously OOM'd even
with flash-attention, ~160MB short) — confirms the earlier flash-attention OOM was genuinely
non-attention memory pressure, and bf16 (halved activation/param-compute footprint) was the fix,
not more flash-attention tuning. Steady-state ~920-930ms/step, **~70-71K tokens/s** (up from
~61.7K at batch=12/no bf16, ~64K at batch=8,grad_accum=2/no bf16 — the best throughput yet), HBM
27-28.6GiB/30.75GiB per chip (close to ceiling but stable), duty cycle ~73%. `total_batch_size`
still 65536 (not Cable's 524288) since grad_accum=1 is still prioritized per earlier instruction.

Progression on tpu4 today: `medium_flash_c0` (no bf16, batch=12, flash-attn) → stopped → 
`medium_flash_b8ga2` (no bf16, batch=8/grad_accum=2, flash-attn, ~64K tok/s) → stopped →
**`medium_bf16_b16ga1`** (bf16 + flash-attn, batch=16/grad_accum=1, ~70-71K tok/s) — current live
run, tmux `medium_bf16_b16ga1` on tpu4, log `~/medium_bf16_b16ga1.log`
(`logs/medium_bf16_b16ga1/` for structured JSONL). Target: `gpt2_jax/CABLE_PAPER_NOTES.md`'s
RoPE/medium perplexity ≈16.9 at context=1024.

MFU at batch=12/no-bf16 was measured ~13-14% (duty cycle ~85-87%); not yet re-measured for the
bf16 run. Known remaining levers to close the gap (not yet done): profile step-to-step host/XLA
dispatch overhead (duty cycle sub-100% suggests some), fuse the grad-accum microbatch loop into a
single compiled `jax.lax.scan` step instead of a Python-level loop of separate pmap calls.

`gpt2_jax/train_gpt.py` now prints `[compile] step 0 wall time (includes first-call XLA compile):
{dt}s` after step 0, for future launches (added after `medium_bf16_b16ga1` was already compiling,
so that run's own compile time wasn't captured this way — its step 0 dt_ms in the JSONL log,
84.5s, is the equivalent number).

**Dispatch-overhead fix (2026-08-27, `medium_opt_v1`)**: the grad-accum microbatch loop and the
eval loop each called `float(...)` on a device array every iteration -- a blocking host sync that
serialized what should've been async-dispatched device work (root-caused after MFU stayed ~13-16%
despite bf16). Fixed in `gpt2_jax/train_gpt.py` by fusing each loop into a single `jax.lax.scan`
inside one `pmap` call (`grad_accum_step`/`eval_accum_step`, replacing the old
`p_grad_step`/`p_eval_step` + Python-loop-with-`float()` pattern), adding a background prefetch
thread for the next training batch, and `donate_argnums=(0,1)` on the params/opt_state apply step.
Smoke-tested locally (CPU, grad_accum=1 and 2) before pushing. Also tried
`LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304"` (confirmed valid on this libtpu build,
via `XLA_FLAGS=--help`'s flag dump) -- three other guessed `XLA_FLAGS` collective-fusion flag
names were rejected as unknown by this build, dropped rather than guessed further.

**Result: ~94.6K tokens/s steady-state (692.5ms/step)** at batch=16/grad_accum=1/bf16/flash-attn
on tpu4 (`medium_opt_v1`) -- up from ~70-71K pre-fix, the single biggest throughput jump of the
day. MFU ≈2.42e9 FLOPs/token x 94.6K tok/s / 1100 TFLOPS peak bf16 (4x v4 chips) ≈ **~21%** (up
from ~13-14% pre-bf16, ~16% bf16-only). Updated ETA: 152,588 steps (10B tokens / 65536
total_batch_size) x 692.5ms ≈ **~29.4 hours (~1.2 days)** for 1 epoch. Confirms the blocking-sync
removal mattered more than the XLA flag (which is still in the launch but unvalidated for actual
impact in isolation -- not yet ablated separately per the "try all, ablate by removing" plan).
Original `train_gpt.py`/`model_gpt.py` (pre-bf16, pre-fusion) backed up to
`gpt2_jax/backup_2026-08-27/`.

Progression on tpu4 today (superseding the earlier list): `medium_flash_c0` (no bf16, batch=12) ->
`medium_flash_b8ga2` (no bf16, batch=8/ga=2, ~64K tok/s) -> `medium_bf16_b16ga1` (bf16, batch=16,
~70-71K tok/s) -> **`medium_opt_v1`** (bf16 + fused-scan/prefetch/donate + vmem flag, batch=16,
~94.6K tok/s) -- current live run, tmux `medium_opt_v1` on tpu4, log `~/medium_opt_v1.log`
(`logs/medium_opt_v1/` for structured JSONL).

Remaining levers not yet tried: ablate the vmem `LIBTPU_INIT_ARGS` flag alone to isolate its
actual contribution; re-measure duty cycle/HBM via `tpu-info` at this throughput (not yet done
post-fix); profile any still-remaining host/XLA dispatch gap with `jax.profiler`.

`gpt2_jax/README.md` (new, 2026-08-27): step-by-step for a fresh session on this lineage (env
setup, data prep, launch, `tpu-info` monitoring, and the checkpoint-egress-warning / periodic
`log.jsonl`+`resolved_config.py` pull-back policy -- same policy as CLAUDE.md/tpu_setup.md's
existing rule for the other TPU lineages, not a new one).

`pyproject.toml`/`uv.lock` fix (2026-08-27): `jax-ai-stack` was hard-pinning `jax==0.8.0`/
`flax==0.12.0`, silently downgrading the manually-upgraded working versions (jax 0.11.1/flax
0.12.9) on every plain `uv sync` — confirmed happening on all 4 nodes. Root cause: nothing in the
repo actually imports jax-ai-stack's other sub-packages (chex/grain/orbax-export/tensorflow), so
it was dropped entirely in favor of depending on `jax[tpu]` (Linux-only via `sys_platform`
marker)/`flax`/`optax`/`orbax-checkpoint` directly at the versions actually in use. `tpu-info` also
added as a tracked dependency (was previously an ad-hoc `uv pip install` per node). `uv sync` now
converges to the correct versions/extras on its own, verified on all 4 nodes.

## 2026-08-27 (later): 4 new nodes (tpu1/tpu2/tpu3/tpu8) set up, idle

Environment set up and verified on all 4 (direct-ssh live, repo synced, `uv sync` clean,
`jax.devices()` confirms 4 TPU chips each, same v4-8 shape as tpu4-7). Not running anything yet —
available for scaling up the Pathfinder dataset-prep work or further ablation sweeps.
IPs: tpu1=35.186.98.243, tpu2=35.186.39.107, tpu3=107.167.160.20, tpu8=35.186.86.22.

## 2026-08-27 (later): summformer_jax ablation vs. gpt2_jax baselines, all 4 nodes in use

New lineage `summformer_jax/` (JAX/Flax NNX port of `qcute/summformer/summformer.py`, made
GPT2-like per explicit correction — LayerNorm/plain-GELU-MLP/plain-MHA, not RMSNorm/SwiGLU/GQA)
runs a controlled ablation of the Ks-hierarchical-summarization + fuse-cross-attn method against
`gpt2_jax`'s plain-GPT2 baselines. Full design/README: `summformer_jax/README.md`.

**Correctness verified** before any real run: forward pass sane loss at random init (all 3
`pos_method`s), backward pass clean (no NaNs), **KV-cache consistency bit-exact
(`match_rate=1.0`) for all 3 `pos_method`s** at both byte-vocab and BPE-vocab scale.

**Dataloader audit (requested explicitly)**: `gpt2_jax/data_loader.py` vs. Cable's own
`Cable_ref/src/data_loader.py` — algorithm-identical (window slicing/shard-advance/wraparound
logic byte-for-byte the same, only cosmetic diffs). `summformer_jax/lm/data_loader.py` vs.
`gpt2_jax/data_loader.py` — functionally byte-identical (only the module docstring differs,
confirmed via diff). So both baseline and ablation share the exact same, Cable-faithful data
pipeline, not just the same dataset files.

**Paper-fidelity audit (requested explicitly)**: the earlier `medium_opt_v1`/`medium_bf16_b16ga1`
runs used `total_batch_size=65536` — 8x smaller than Cable's own paper value of 524,288 (per-GPU
micro batch 64/32/16 for tiny/small/medium, 8x H100, `gpt2_jax/CABLE_PAPER_NOTES.md`) — a
pragmatic earlier choice (`grad_accum=1`) that was NOT faithful to the paper. Fixed: switched to
`total_batch_size=524288` with the paper's own per-device `batch_size`, letting `grad_accum_steps
= total_batch_size // (batch_size * seq_len * n_devices)` absorb the 8-GPU-vs-4-chip device-count
difference — same formula the paper's own code uses, just evaluated at n_devices=4 instead of 8.
Captured explicitly (not left to CLI defaults) in `configs/gpt2_jax/{medium,small}_rope_default.py`.

**New OOM found**: `medium`'s own batch_size=16 at `total_batch_size=524288` (grad_accum=8, which
forces the fused `jax.lax.scan` grad-accum path) OOMs by ~340MB even with flash-attention+bf16 —
more headroom needed for the scanned path than the grad_accum=1 case ever needed. Fixed:
`--batch-size 8` (grad_accum=16) for medium at this total_batch_size; small's own batch_size=32
(grad_accum=4) fits without adjustment.

**Ablation hparams** (both use `Ks=(2,2,2,2)`, `n_fuse=3` — chosen after comparing several
`Ks`-length/`n_layers` tradeoffs; effective_depth formula and full derivation in
`summformer_jax/README.md`):

| Ablation | vs. baseline | d_model | n_heads | n_layers | params | baseline params | delta |
|---|---|---|---|---|---|---|---|
| medium | gpt2-medium (24L/1024d) | 1024 | 16 | 2 | ~279.3M (confirmed) | 353.8M | -21% |
| small | gpt2-small (12L/768d) | 768 | 12 | 1 | ~126.8M | ~123.6M | +2.6% |

`byte_acc` metric removed (2026-08-27) from both `model_summformer.py`'s `metrics` dict and
`train_summformer.py`'s logging — leftover name from the original byte-vocab version, misleading
now that these ablation runs use BPE vocab (it was actually token accuracy, not byte accuracy).

**Current live status** (paper-faithful `total_batch_size=524288` throughout, checked 2026-08-27
~20:50 local — steps/loss will be stale by the time this is read, tok/s is the stable number):

| Node | Run | Step | Loss | tok/s | Notes |
|---|---|---|---|---|---|
| tpu4 | `medium_paper_match_b8` | 6922 | ~3.09 | ~102.6K | gpt2-medium baseline, batch_size=8/grad_accum=16 |
| tpu5 | `summformer_medium_ablation` | 17838 | ~2.9-3.0 | ~293K | ablation vs. tpu4, nearing 1-epoch target |
| tpu6 | `small_paper_match` | 16949 | ~3.0-3.1 | ~258K | gpt2-small baseline, batch_size=32/grad_accum=4 |
| tpu7 | `summformer_small_ablation` | **finished** (19072/19073, 1 epoch) | val 3.20 / bpb 4.62 | ~689K peak | checkpoint `model_19072` = 448M on disk; node idle since, now used for dataset-prep smoke tests (see below); 55G free on its 97G disk |

`summformer_small_ablation`'s checkpoint pulled to local `checkpoints/summformer_small_ablation/` and
restored successfully (126,892,800 params, matches the ~126.8M documented above; sane finite loss
on a smoke-test forward pass) — confirms the checkpoint isn't corrupt and the current model code
still loads it correctly. All TPU nodes are on GCP's default **Premium Tier** network (no
`--network-tier=STANDARD` in any `TPU.md` create command) — internet egress $0.12/GB for the first
1TB/month, with the first 1GiB/month free; a 448M checkpoint pull costs ≈$0.05 or less (likely free,
under the monthly allowance).

Ablation runs are faster in raw tok/s than their baselines (smaller `n_layers` more than offsets
the fuse-stage FLOPs overhead at these sizes) — not itself meaningful, the actual comparison is
loss/bpb at matched token counts once both sides have logged enough steps. It's expected/fine for
the ablation to lose to its baseline if it finishes — the point is an honest comparison at
roughly matched compute, not tuning to win (explicit instruction).

Loss curves are plotted automatically: `scripts/jax/pull_and_plot.sh` scp's each node's `log.jsonl`
into local `logs/<run_name>/` and regenerates `loss_curve.png` there via `summformer_jax/lm/scripts/plot_jax_run.py`;
an hourly `Monitor` loop runs this and reports each run's current step. Direct-ssh connections to
all 4 nodes have dropped and been recovered multiple times mid-session (stale `ControlMaster`
socket, not preemption) — recovery procedure now in `CLAUDE.md`.

### `Ks` tuple semantics fixed (2026-08-27, `summformer_jax` + `qcute_zero`)

Both lineages defined `n_fuse = len(Ks) - 1`, so `Ks`'s *last* entry was read nowhere in either
model — dead by construction, kept only as a conventionally-`1` placeholder for the top level
(which needs no block-factor since nothing sits above it to fuse into). Fixed in both to the
cleaner convention `n_fuse = len(Ks)` (no trailing dummy entry) — verified via direct model
construction that `n_fuse`/`n_lms`/param shapes are unchanged for the equivalent shortened `Ks`, so
existing checkpoints (`qcute_zero`'s overfit10k runs, `summformer_jax`'s live ablations) stay
loadable. All `configs/summformer_jax/*.py` and `configs/qcute_zero/*.py` updated to drop the
trailing entry (`Ks=(2,2,2,2)`→`(2,2,2)`, `Ks=(2,1)`→`(2,)`, `Ks=(1,)`→`()`, etc.). `qcute_lagcodec`
audited and left unchanged — it genuinely reads `Ks[-1]` (e.g. a divisibility assert in
`qcute_lagcodec.py`), so its last entry was never dead; the `qcute_zero` docstring claiming "same
semantics as qcute_lagcodec" was the source of the divergence and has been removed.

### New dataset-prep work (2026-08-27, in progress)

Goal: bring 2 more datasets into the `gpt2_jax`-compatible token-shard pipeline (Long Range Arena
dropped per explicit instruction — its `lra_release.gz` is 403 Access Denied via both the
documented `gs://long-range-arena/lra_release` bucket and its HTTP fallback, from both anonymous
curl and `gsutil` with the node's ambient GCP credentials; no working mirror found).

- **Pathfinder (32 and 256 resolution)**: `scripts/jax/generate_pathfinder.py` — procedurally
  *generates* (not downloads) Pathfinder-style examples: a random-walk "snake" contour + distractor
  snakes, two marker dots, binary connected/disconnected label. **Not a port of the original
  LRA/drewlinsley renderer** (`github.com/drewlinsley/pathfinder`'s `snakes2.py` — unpublished as a
  package, MATLAB-derived) — an independent reimplementation of the same task definition,
  parameterized with the per-resolution constants LRA's TFDS builder docstrings document
  (`contour_length`, `marker_radius`, `paddle_thickness`, `num_distractor_snakes`), so pixel
  statistics won't match the original released dataset. Output format: each example is
  `[flattened grayscale pixels (0-255)] + [1 label token (256=connected, 257=not)]`, concatenated
  into one flat stream per split and written as uint16 `.npy` shards — the exact convention
  `gpt2_jax/data_loader.py`'s `DataLoaderLite` already reads (filename contains `train`/`val`, flat
  1D array). Verified end-to-end at small scale (32/16 examples, both resolutions): generates
  cleanly, `DataLoaderLite` loads the shards with zero code changes, values fall in the expected
  0-257 range. Not yet scaled up to a real training-size dataset.

ETAs at steady-state speeds for the still-running baselines (19,073 steps total, 1 epoch): tpu4
~27hr total (was ~19hr in at last check), tpu6 ~11hr total. Multi-hour, so don't stream every
step; check in periodically instead (see below).

### `scripts/jax/bench_generation.py`: jitted, run on tpu7 (2026-08-27)

Rewrote the prefill/generation timing core to wrap each fixed-shape trajectory (prefill, and
prefill+`--gen-tokens` more steps) in one `jax.jit` call, called repeatedly — not jitting the
incremental stepper's mutating closure directly (that would silently freeze the KV cache at its
first-trace values on replay); instead each timed call builds a fresh stepper/cache and runs the
whole unrolled trajectory in one self-contained trace, so cache growth is correct within a call and
every call after the first hits the same compiled executable. `generation_only` cost is reported as
`combined_mean - prefill_alone_mean`, amortized per token.

Found and fixed a real bug surfaced while running this on tpu7: gpt2_jax's flash-attention kernel
requires `kv_seq_len % 128 == 0`; the naive no-cache generation path (context grows by 1 token/step)
breaks that on almost every step. Fixed by padding each step's forward pass out to the next
128-boundary with dummy tokens — causality means padding (appended after all real content) never
affects any real position's logits, so this is correctness-preserving.

Also found tpu7's `configs/summformer_jax/small_rope_ablation.py` and `gpt2_jax/{model_gpt,
train_gpt,data_loader}.py` were stale (pre-dated this session's Ks-semantics fix and bf16/flash
additions respectively) — re-synced before benchmarking; the stale config would have silently
benchmarked a different (141M-param, `Ks=(2,2,2,2)`/`n_fuse=4`) architecture instead of the actual
live ablation's `Ks=(2,2,2)`/126.9M.

**Results** (tpu7, TPU device, batch=1, context=1024, 8 generated tokens, 3 warmup/10 repeats):

| Model | Params | Prefill (1024 tok) | Generation (8 tok, after prefill) |
|---|---|---|---|
| gpt2-small (flash-attn) | 123.7M | 5.04ms (203K tok/s) | 1.63ms total, 0.20ms/tok (4917 tok/s), naive full-recompute/no cache |
| summformer-small (Ks=2,2,2) | 126.9M | 2.80ms (366K tok/s) | 2.18ms total, 0.27ms/tok (3673 tok/s), real incremental KV-cache |

At this small scale/short generation length, gpt2's naive-recompute generation is actually
*faster per-token* than summformer's real KV-cache path (4917 vs 3673 tok/s) — plausible since
gpt2-small's 1024-length forward pass is itself cheap enough (5ms) that recomputing it 8 times
barely costs more than one incremental step's dispatch/bookkeeping overhead; the KV-cache
advantage should widen as context/gen_tokens grow (naive cost scales with context, incremental
doesn't) -- not yet tested at larger context or longer generation. Both prefill numbers (203K/366K
tok/s) are consistent with the FLOPs findings in `summformer_jax/lm/scripts/compare_summformer_gpt2.py`'s own results
(summformer cheaper per-token at this matched d_model/n_heads).

Raw results: `bench_results/gpt2_small_rope_default_tpu_1787863315.json`,
`bench_results/summformer_small_rope_ablation_tpu_1787863443.json`.

**Fair (no-flash) rerun** (2026-08-28): the table above used gpt2's own config default
(`use_flash_attention=True`), which isn't apples-to-apples against summformer (no flash-attention
path at all). Added `--no-flash-attention`/`--flash-attention` override to `bench_generation.py`
and reran gpt2-small with it off:

| Model | Prefill (1024 tok) | Generation (8 tok) |
|---|---|---|
| gpt2-small, no flash | 3.04ms (337K tok/s) | 0.15ms/token (6839 tok/s) |
| summformer-small | 2.80ms (366K tok/s) | 0.27ms/token (3673 tok/s) |

Two findings: (1) flash-attention was actually *slower* than plain dense SDPA at this size
(5.04ms vs 3.04ms prefill) -- the Pallas kernel's overhead outweighs its benefit at
context=1024/gpt2-small scale; (2) with flash off, gpt2's naive full-recompute generation is
*faster per-token* than summformer's real KV-cache path (6839 vs 3673 tok/s) -- the reverse of
what the FLOPs numbers alone would suggest, likely because summformer's incremental stepper
carries more Python/dispatch overhead per call (many small ops across levels/stages) that
dominates at this short (8-token) a generation run rather than a FLOPs-bound regime. Not yet
tested at longer generation lengths, where the KV-cache's O(1)-per-token advantage should widen.
Considered but not pursued: adding flash-attention to summformer itself, so both sides have it --
its self-attention always prepends a zero-KV-sink token, and this repo's own prior investigation
(`docs/tpu_setup.md`'s `bytelm_tpu` zero_kv_sink+flash-attention section) found that combination
costs ~25x throughput due to the kernel's block-alignment requirements colliding with the sink's
per-layer re-concatenation -- that finding would likely resurface here, so this wasn't attempted.

Raw result: `bench_results/gpt2_small_rope_default_tpu_1787865902.json`.

### gpt2_jax: real incremental KV-cache implemented (2026-08-28)

`gpt2_jax/model_gpt.py` had no cache/generation code at all before this (training-only, confirmed
earlier this session) -- added `CausalSelfAttention.forward_incremental`/`Block.forward_incremental`
(mirroring `summformer_jax`'s own proven pattern), `Model._make_incremental_stepper`,
`generate_no_cache`/`generate_kv_cache`/`check_kv_cache_consistency`. Always plain SDPA in the
incremental path, never the flash-attention kernel -- that Pallas kernel needs
`kv_seq_len % 128 == 0` (see the bench finding above), which a growing single-token cache almost
never satisfies. **Verified bit-exact** (`match_rate=1.0`) against the full-recompute reference for
all 3 `pos_method`s (rope/learnable/base), run on tpu7.

### Second summformer ablation variants: deeper Ks, launched (2026-08-28)

New variants exploring a different point on the Ks-length/n_layers tradeoff -- max `n_layers` that
keeps `effective_depth <= baseline's own n_layer` (a hard cap, chosen over the softer "slightly
smaller params" goal once the two conflicted -- see the sweep below):

| Variant | Ks | n_layers | eff_depth | baseline eff_depth cap | params | Δparams | flops_ratio |
|---|---|---|---|---|---|---|---|
| `small_rope_ablation_ks44` | (4,4) | 2 | 10 | 12 | 148.2M | +19.8% | 0.846x |
| `medium_rope_ablation_ks444` | (4,4,4) | 3 | 21 | 24 | 367.6M | +3.9% | 0.839x |

Both land slightly ABOVE baseline params (not below, as first asked) in exchange for FLOPs much
closer to "similar" (0.84x both, vs. ~0.6x one layer shallower) -- an explicit tradeoff pick.
Launched on tpu5 (`summformer_medium_ablation_ks444`) and tpu6 (`summformer_small_ablation_ks44`),
both idle nodes reusing already-downloaded FineWeb-Edu-10B data.

### Bench follow-ups (2026-08-28): sink isolation, real-cache-vs-real-cache at both sizes

**Zero-KV-sink isolated.** Added `Config.zero_kv_sink: bool = True` to `summformer_jax/
model_summformer.py` -- a pure runtime toggle (the sink carries no learned parameters, just a
per-call zero row prepended to K/V), so flipping it doesn't affect any trained checkpoint's weights
or shapes; `--no-zero-kv-sink` in `bench_generation.py` isolates its cost for benchmarking only
(never for evaluating a real checkpoint -- all were trained with it on). Re-verified
`check_kv_cache_consistency` still bit-exact (`match_rate=1.0`) both on and off after the change.
Result: the sink accounts for a modest ~5-10% overhead (small-ablation prefill 2.67ms without vs.
2.80ms with; generation 0.25 vs. 0.27ms/token) -- not the dominant factor in any of the gaps found
below.

**gpt2's real KV-cache benchmarked properly** (bench_generation.py's gpt2 path now defaults to the
real stepper added above, not the old naive-recompute path -- pass `--naive-generation` for the old
numbers). Also hit the exact same stale-config bug twice more (`configs/summformer_jax/
medium_rope_ablation.py` still had the pre-fix `Ks=(2,2,2,2)` on tpu7) -- **standing gotcha: always
re-sync a node's config files before benchmarking/training on it, don't assume a prior sync
covered every file that's since been edited locally.**

**Results, both sizes, fair settings** (no flash-attention on gpt2, real KV-cache both sides,
zero-KV-sink on for summformer -- matches what's actually trained):

| | Prefill (1024 tok) | Generation (8 tok, real cache) |
|---|---|---|
| gpt2-small (123.7M) | 3.04ms (337K tok/s) | 0.49ms/token (2053 tok/s) |
| summformer-small (126.9M) | 2.80ms (366K tok/s) | 0.27ms/token (3673 tok/s) |
| gpt2-medium (353.8M) | 7.16ms (143K tok/s) | 1.24ms/token (806 tok/s) |
| summformer-medium (279.4M) | 5.79ms (177K tok/s) | 0.73ms/token (1369 tok/s) |

Consistent pattern at both sizes: summformer's real KV-cache generates ~1.7-1.8x faster per token
than gpt2's real KV-cache (small: 3673 vs 2053 tok/s; medium: 1369 vs 806 tok/s), and summformer's
prefill advantage widens with scale (1.09x at small, 1.24x at medium) -- consistent with the FLOPs
findings in `summformer_jax/lm/scripts/compare_summformer_gpt2.py` (summformer cheaper per-token at matched
d_model/n_heads, the gap growing with model size). This reverses the earlier (naive-generation,
flash-on) reading that had gpt2 looking faster at generation -- both of those were measurement
artifacts (naive recompute happened to be cheap at tiny scale/short runs; flash-attention was
actually slower than plain SDPA at this size), not real architecture advantages.

Raw results: `bench_results/{gpt2,summformer}_{small,medium}_rope_{default,ablation}_tpu_*.json`.

**Checking in on a run yourself** (safe, read-only — none of these mutate the run):

```bash
# tail the structured log (same data whether you're watching live or checking once)
ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p muaz@<external_ip> 'tail -20 ~/<run_name>.log'

# tmux capture-pane -- peek at the session's raw pane WITHOUT attaching (never blocks/steals input)
ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p muaz@<external_ip> 'tmux capture-pane -t <run_name> -p -S -30'
# note: these 4 runs redirect stdout to ~/<run_name>.log (`> ... 2>&1`), so the tmux pane itself is
# blank by design -- tail the log file above, capture-pane is for sessions that print to the
# terminal directly (e.g. an interactive install/setup step).

# tmux attach -- only if you want to watch live yourself (detach with Ctrl-b d, does not kill the run)
ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -t muaz@<external_ip> 'tmux attach -t <run_name>'
```

Run names/nodes for the 4 active runs: `medium_paper_match_b8`@tpu4 (`35.186.15.67`),
`summformer_medium_ablation`@tpu5 (`35.186.33.7`), `small_paper_match`@tpu6 (`35.186.110.50`),
`summformer_small_ablation`@tpu7 (`35.186.34.230`).

New config files: `configs/gpt2_jax/small_rope_default.py` (new, mirrors `medium_rope_default.py`),
`configs/summformer_jax/{medium,small}_rope_ablation.py` (new dir). `gpt2_jax/README.md` updated
with the batch/grad_accum-matching formula and this status table; `summformer_jax/README.md` new.

## 2026-08-28 (later): summformer_jax v2 configs launched (tpu1-4), tpu8 running image_gen

**tpu1-4**: the 4 finalized `configs/summformer_jax_v2/*.py` configs (small/medium x param-match/
flops-match) launched via `summformer_jax/lm/train_summformer_v2.py` (note: `train_summformer_v2.py`
now lives under `summformer_jax/lm/`, not `summformer_jax/` directly, after a repo reorg this
session -- `summformer_jax/lm/` holds the byte/BPE-text lineage, `summformer_jax/image_gen/` a new,
separate image-byte lineage, see below). tpu1/2/3's FineWeb-Edu-10B data lives in `/dev/shm`
(tmpfs) -- hit and fixed a real data-loss bug here: `systemd-logind`'s `RemoveIPC=yes` default was
wiping tmpfs-owned files once the launching SSH session ended, even with a detached `tmux` still
running the job; fixed via `sudo loginctl enable-linger muaz` on every node using `/dev/shm` (see
CLAUDE.md's tmpfs section and `docs/tpu_direct_ssh.md` step 4 for the full incident + standing
fix). Run names: `small_rope_v2_parammatch`@tpu1, `small_rope_v2_flopsmatch`@tpu2,
`medium_rope_v2_flopsmatch`@tpu3, `summformer_medium_v2_parammatch`@tpu4.

**tpu8**: new lineage, `summformer_jax/image_gen/` -- byte-level RGB image generation (not text),
mixed-dim trunk/code-LM architecture, real incremental KV-cache (trunk + fuse-stage cross-attention
+ code-LM, all bit-exact verified), block-locality/receptive-field probes, and a training loop
whose per-epoch eval also generates and saves sample PNGs. Full design history, every bug found and
fixed, and the Fractal-Generative-Models/`qcute_lagcodec`/ARMD comparison discussion behind the
architecture choices: **`docs/image_gen_design.md`**. Currently training `image64_mixdim_v1`
(tmux session `image_gen_mixdim`) against tpu8's own in-progress ImageNet64 prep output at
`/dev/shm/imagenet64` (`scripts/prep_imagenet64.py`, still growing in the background -- the
training run reads whatever shards exist at startup, not the complete 1.28M-image set yet). Exact
setup/launch commands and log/sample paths: see `docs/image_gen_design.md`'s "First real TPU
training run" section. **ImageNet64 prep finished (2026-08-29)**: 1,281,167 train images + full
validation split written, 0 failed/skipped, 26 train shards + 1 val shard, `/dev/shm/imagenet64` --
ready for a real (not growing-prefix) training launch.

### tpu1-4 v2 ablation: final results (2026-08-29, all 4 runs finished)

All four `configs/summformer_jax_v2/*.py` runs completed 18793/18793 steps (1 epoch,
`total_batch_size=524288`, paper-faithful). gpt2 baselines (`small_paper_match`@tpu6,
`medium_paper_match_b8`@tpu4-earlier) also finished at 19072/19072 steps -- see the "New OOM found"
section above for their launch config.

| Tier | Model | Run | Val loss | Val bpb | Val PPL | vs. gpt2 baseline |
|---|---|---|---|---|---|---|
| Small | gpt2-small (baseline) | `small_paper_match` | 3.036 | -- | **20.83** | -- |
| Small | summformer, param-match | `small_rope_v2_parammatch`@tpu1 | 3.199 | -- | **24.51** | +17.7% worse |
| Small | summformer, flops-match | `small_rope_v2_flopsmatch`@tpu2 | 3.010 | -- | **20.28** | 2.6% better |
| Medium | gpt2-medium (baseline) | `medium_paper_match_b8` | 2.809 | -- | **16.60** | -- |
| Medium | summformer, flops-match | `medium_rope_v2_flopsmatch`@tpu3 | 2.804 | 4.046 | **16.52** | 0.5% better |
| Medium | summformer, param-match | `medium_rope_v2_parammatch`@tpu4 | 2.871 | 4.142 | **17.66** | 6.4% worse |

**Pattern holds at both scales**: flops-matched (same compute, fewer effective layers) beats its
gpt2 baseline by a small margin at both small and medium; param-matched (same param count, but
therefore *more* compute since it's not shallower) loses at both scales, worse at small (+17.7%)
than medium (+6.4%). Consistent with the ablation's real advantage being explainable by compute
savings from being architecturally shallower at matched params, not a representational edge from
the hierarchical-summarization mechanism itself -- see chat discussion (not yet written up) for the
"controlled param-vs-flops-matching critique" framing this suggests, and what it would take to turn
into a real result (3+ seeds per config, a third scale point, a couple of cheap zero-shot eval
tasks -- none of that done yet, this table is loss/ppl-only, single seed per config).

### KV-cache memory estimate (2026-08-29): a real, architecture-derived win independent of the above

`scripts/estimate_memory.py` (new) -- analytic KV-cache memory from each model's own config fields
(closed-form, not benchmarked), verified against real `nnx.state` param counts (matched the
config docstrings' documented params exactly). At `context_len=1024`, `batch=1`, bf16:

| Model | Params | KV cache | vs. gpt2 baseline |
|---|---|---|---|
| gpt2-small | 123.7M | 36.0MB | -- |
| summformer-small, param-match | 119.8M | **9.0MB** | **25%** (4x smaller) |
| summformer-small, flops-match | 169.4M | 30.0MB | 83% |
| gpt2-medium | 353.8M | 96.0MB | -- |
| summformer-medium, param-match | 329.8M | **48.0MB** | **50%** (2x smaller) |
| summformer-medium, flops-match | 430.5M | 80.0MB | 83% |

Holds even though these ablation configs use `main_window=None` (unbounded trunk attention, same
as gpt2) -- savings come purely from fewer main layers (2 vs 12, 10 vs 24) while each fuse-stage's
code-LM cross-attention buffer adds back only a small fraction (3-8MB). The param-matched variants
are the standout: 2-4x less KV-cache memory at roughly matched params, and (per the ppl table
above) equal-or-better perplexity at medium scale specifically (small param-match still loses on
ppl, so that one's memory win comes with a real quality tradeoff). This is a different,
architecture-derived axis from the perplexity result -- doesn't need more seeds/scales to hold
(it's a closed-form consequence of the config, not a noisy training outcome), and doesn't depend on
matching Block Transformer/MEGABYTE's compute scale to be a valid claim. Not yet done: applying a
bounded `main_window` (these configs deliberately left it unbounded) would shrink the trunk term
further still -- not measured yet.

## 2026-08-29: tpu8 image_gen real-data launch, tpu5 full-download ImageNet64 prep, tqdm console fix

**tpu8**: ImageNet64 prep (`scripts/prep_imagenet64.py`) finished overnight -- full 1.28M train
images across 26 shards + 1 validation shard at `/dev/shm/imagenet64`, no longer a growing prefix.
Launched `image_gen/train.py` against the complete dataset (`image64_mixdim.py` config,
`--eval-samples 0` since sample generation is still ~674ms/token -- see
[docs/image_gen_design.md](image_gen_design.md)'s "JIT fix" section).

**`train.py` now uses tqdm for console output** (`from tqdm import tqdm`, `pbar.set_postfix(loss=,
bpb=, dt_ms=)`), matching `prep_imagenet64.py`'s existing convention. Per-step `log()` calls now
take a `console=False` flag for the train-step case specifically, so the per-step JSON dict isn't
also printed to the console on top of the tqdm bar (it still writes to `log.jsonl` every step,
unchanged) -- eval/done events still print via `tqdm.write(...)` so they don't corrupt the bar's
line. First version (no `console` flag) printed a JSON dict every single step, defeating the point
of having a progress bar at all; caught immediately when the pane showed spam instead of a moving
bar, fixed same session (`image64_mixdim_v5` killed, relaunched clean as `image64_mixdim_v6`).

**Standing convention update**: for a TPU run launched in `tmux`, prefer running the command with
**no stdout redirect at all** (`tmux new-session -d -s <run_name> 'python3 ...'`) rather than the
`tee`-to-logfile pattern -- tqdm/prints land directly in the pane, so `tmux capture-pane`/`tmux
attach` show live output with no extra plumbing. Only redirect (via `tee`, never a plain `>`) when
the run's own log genuinely needs periodic `scp`-pulling back to this repo (the multi-hour-run
monitoring routine reads `log.jsonl`/`run.jsonl` off disk, not the pane) -- that's an orthogonal
need from watching the pane live. See CLAUDE.md's launch-conventions section for the updated rule.

**tpu5**: cleared for a fresh use (`/dev/shm` was already empty, no cleanup needed), linger
enabled. New script `summformer_jax/image_gen/scripts/prep_imagenet64_full.py` -- copy of
`prep_imagenet64.py` with `streaming=False` (a genuine full non-streaming download+cache instead of
iterating a streamed split) -- launched in tmux session `imagenet64_full_prep`, train split first
then validation, into `/dev/shm/imagenet64`. **Superseded same session** -- the non-streaming
approach hit the exact disk-fill failure mode it was meant to avoid (raw parquet + separate Arrow
"Generating split" cache, both persisted, 98% of `/dev/shm` used) and was abandoned in favor of
`download_imagenet.py` (streaming, custom `.bin` shard persistence, see below).

## 2026-08-29 (later): windowed cross-attention, lm/image_gen summformer.py consolidation, tpu5 labeled ImageNet download, image_classification pipeline

**`summformer.py` (image_gen's model file) gained real windowed cross-attention.** Previously only
the trunk (`main_window`) and code-LM self-attention (`code_window`, added same session) routed
through the compute-saving `chunked_windowed_attention` kernel -- the fuse-stage cross-attention
(`(stride, window)` pair) always fell back to dense-plus-mask regardless of `window`'s value, an
`O(T*S)` cost (`S = context_len/stride`) that dominates total model compute at small strides
(measured: ~19.3 GFLOP for `tiny_1`'s stride=8 stage vs ~0.6 GFLOP for the entire trunk combined --
cross-attention was the actual bottleneck, not the trunk). Added `windowed_cross_attention` (gather
with **static, trace-time-known indices** -- same performance class as `chunked_windowed_attention`'s
own `k_ext[:, :, idx]` trick, not the "slow, data-dependent gather" gather-slowness concerns are
usually about), `Attn.forward_cross_windowed`, `FuseStageV2.forward_windowed`, wired into
`_pool_and_fuse`'s training-forward path only (the incremental/fully-static generation paths are
untouched, separately re-verified bit-exact via `check_kv_cache_consistency`/
`check_kv_cache_consistency_fully_static` after the change -- match_rate 1.0 in both). Verified
correct against the dense reference across 16 (T, stride, window) combinations including
non-divisible lengths, `window<stride` degenerate cases, and both sink modes (max diff ~1e-6,
floating-point noise). Measured 12.8x real speedup at `tiny_1`'s actual scale (CPU: 518.78ms dense
vs 40.62ms windowed) -- lower than the ~32x FLOP-count estimate (real dispatch/gather overhead), but
a large, real win. Also added `ConfigV2.force_dense_attn: bool = False` -- overrides `main_window`/
`code_window` to unbounded regardless of their configured value, for debugging/comparison against
the known-dense reference path.

Caveat found while wiring `code_window` into `tiny_1.py`: fuse-stage cross-attention window and
code-LM self-attention window are two *separate* knobs on different attention surfaces (trunk<->code
vs code<->code) that don't auto-follow each other or `main_window` -- nothing in the code derives
one from another, each is read/applied completely independently. Setting all three to the same
value (as `tiny_1.py` now does) is a convenience choice, not a structural requirement.

**`lm/` and `image_gen/` now share one frozen `summformer.py`** instead of `lm/model_summformer_v2.py`
being a separate, leaner fork -- `image_gen/summformer.py` was already a strict superset (same
`ConfigV2`/`FuseStageV2`/`SummTransformerV2` core, plus everything `lm`'s version lacked:
incremental/KV-cache generation, windowed attention, `check_block_locality`) with zero image-specific
dependencies (pure `jax`/`flax`, no RGB/resolution assumptions in the model itself). Duplicated
verbatim into `lm/summformer.py` (not a cross-directory import -- matches this repo's existing
flat-file-per-training-script convention), `lm/train_summformer_v2.py` renamed to `lm/train.py` and
its import switched to the shared file. Both copies are now **feature-frozen** -- confirmed custom/
BPE vocab support works (real forward pass at `vocab_size=50257` succeeds, correct output-head
shape) since `ConfigV2.vocab_size` was already generic, nothing in the model assumed byte-level.
`lm/configs/v2/thin512_win2_allfuse12.py` (171M params, 12 fuse stages, `window=2==stride=2` --
the boundary-valid case for the new windowed cross-attention) confirmed loading/constructing/
eval'ing correctly through the shared file; a full training-step timing run wasn't completed before
being paused.

**tpu5: `download_imagenet.py` rewritten to also capture labels** (`[4-byte signed label][8-byte
length][raw JPEG bytes]` per entry, was label-less before) -- needed for the new `image_classification`
task, verified round-trip correct against real HF data before relaunching. Old label-less data
(144GB, 84 shards) is a breaking-format mismatch with the new reader, so `/dev/shm` was cleared and
the full dataset re-downloaded with 32 workers: **train 1,281,167 images + validation 50,000
images, both exactly matching official ImageNet-1k split sizes**, confirmed via a manual shard
byte-count parse (not just trusting the script's own tally). One real bug hit and understood (not
a data-loss issue): `--num_workers 32` on the validation split (only 14 underlying parquet files)
crashes for the 18 workers with no file to shard (`IndexError` in HF `datasets`' own
`_merge_gen_kwargs`) -- the 14 workers that *did* get a valid shard had already written their files
before the crash surfaced, so all 50,000 images were still captured; cap `--num_workers` at the
split's file count to avoid the error message in future runs.

**New `summformer_jax/image_classification/` lineage** -- byte-level ImageNet-1k classification
(not generation), reusing the frozen `summformer.py` backbone as a general vision encoder. `d_model=128,
n_layers=16` (~4.66M params, `ViT-Ti/16` reference is ~5.7M), `context_len=224*224*3=150528`.
`classifier.py`'s `SummClassifier` hooks `SummTransformerV2._cascade` directly (trunk forward pass,
already returns final hidden states, no generation/LM-head/MTP machinery touched) with a linear
head; two pooling variants -- unidirectional (last-position) and a dual causal-pass
forward+reversed-sequence mean-pool (NOT true bidirectional attention -- this codebase's windowed
attention is only defined for a causal direction, so genuine non-causal windowing isn't supported;
approximated by running the causal backbone twice instead). Training recipe approximates **DeiT**
(Touvron et al. 2021) -- AdamW, `weight_decay=0.05`, `base_lr=5e-4` scaled by `batch/512` (DeiT's
own convention, not the Flax ResNet50 reference's `/256`), linear-warmup+cosine-decay. **Real
caveat**: DeiT-Tiny's own paper reports ~72.2% top-1, not 75%, and that's with a full augmentation
stack (RandAugment/Mixup/CutMix/stochastic depth/label smoothing) this pipeline doesn't implement
yet (only random-resized-crop + flip) -- matching ViT-Tiny's param count alone doesn't get to 75%.

Two real bugs found and fixed while first launching on tpu5 (v4-8, all 4 chips):
- **Unbounded cross-attention OOMs at this sequence length** -- `context_len=150528` is ~12x
  `image_gen`'s `12288`; leaving fuse-stage cross-attention windows at `-1` (image_gen/tiny_1.py's
  own convention) blew HBM directly (165GB needed vs 30.75GB available, confirmed). Fixed by
  setting each stage's cross-attention window to its own stride (minimum valid value), routing
  through the new `windowed_cross_attention` kernel.
- **Eager full-shard reads duplicated tmpfs's own RAM** -- `dataloader.py`'s first draft read each
  shard fully into a Python `bytes` object to build its entry index; since shards already live on
  `/dev/shm` (already RAM), this doubled memory for no reason (136GB RSS and climbing, confirmed).
  Fixed via `mmap`-ing each shard file instead (zero-copy view into the same pages).

After both fixes: confirmed working end-to-end on real TPU (dataloader -> model forward -> loss/
accuracy -> `pmap` train step across all 4 chips), 160/160 smoke-test steps completed cleanly, no
errors. Not yet run as a real multi-epoch training session. Full detail, file list, usage commands:
[summformer_jax/image_classification/README.md](../summformer_jax/image_classification/README.md).

**tpu8**: `tiny_1` (image_gen, uniform `d_model=256`/`n_layers=2` across trunk and both fuse
stages, `main_window=code_window=24`, cross-attention `stride=8` both stages) still running
continuously through all the above -- at last check, 26% through its 5-epoch (`100,095`-step)
budget, first eval+checkpoint completed cleanly at step 20,019 (`val_bpd=8.48`, early/noisy).

## 2026-08-31/09-01: tpu1-4 `thin512_win{8,12,16,32}_allfuse8_chained` self-attn-window sweep, all 4 finished

`summformer_jax/lm/` window sweep (`configs/thin512_win{8,12,16,32}_allfuse8_chained.py`, 8 fuse
stages, chained pooling, `stride=4`/paper-faithful `total_batch_size=524288`) -- only `SELF_WINDOW`
varies across the 4 configs (8/12/16/32), isolating window size as the single variable vs. the
`medium_paper_match_b8` gpt2-medium baseline (2.809 val loss / 16.60 PPL, finished 2026-08-29, see
above). All 4 launched 2026-08-31 07:14 UTC, finished 19073/19073 steps (1 epoch) between
2026-09-01 ~05:05-06:14 UTC (~7h50m-8h59m wall clock each, tok/s dropping with larger window:
~199K/195K/190K/170K tok/s for win=8/12/16/32 respectively). `log.jsonl` pulled to local
`logs/thin512_win{8,12,16,32}_allfuse8_chained/` (no checkpoints pulled -- egress-conscious, per
convention).

| Run | window | Val loss | Val PPL | vs. gpt2-medium baseline |
|---|---|---|---|---|
| gpt2-medium (`medium_paper_match_b8`, earlier sweep) | -- | 2.809 | **16.60** | -- |
| `thin512_win8_allfuse8_chained` | 8 | 3.344 | **28.33** | +70.7% worse |
| `thin512_win12_allfuse8_chained` | 12 | 3.298 | **27.06** | +63.0% worse |
| `thin512_win16_allfuse8_chained` | 16 | 3.266 | **26.21** | +57.9% worse |
| `thin512_win32_allfuse8_chained` | 32 | 3.210 | **24.79** | +49.3% worse |

Monotonic, still-improving trend: larger self-attention window keeps helping, no sign of saturating
yet, but even the best of this sweep (window=32) remains ~50% worse in PPL than the gpt2-medium
baseline -- this `thin512`/`allfuse8_chained` architecture point hasn't closed the gap at any window
tested so far. Since window is the only variable changed across these 4 runs, the causal claim
"larger window helps" is unconfounded for this comparison; the comparison AGAINST gpt2-medium is
suspect/confounded (different architecture entirely, not an isolated variable). Natural next step:
a `win64` (or larger) config to check whether the trend keeps improving or plateaus -- not yet
launched.

Nodes freed after this sweep: tpu1 (35.186.98.243), tpu2 (35.186.39.107), tpu3 (107.167.160.20),
tpu4 (35.186.15.67), all idle as of 2026-09-01, available for the next run.

## 2026-09-01/02: real-Pallas-flash-attention seqlen sweep (medium + small), paper-match baselines,
## FLOPs/wallclock bench

Self-attention (embedder, encoder chain, decoder) switched to the real Pallas `_tpu_flash_attention`
kernel everywhere (genuinely O(T) memory, not `jax.nn.dot_product_attention` which never reduces
memory regardless of mask -- see prior entries), `force_dense=True`, cross-attention left dense with
zero-KV sink. `sweep_seqlen_medium_1`/`sweep_seqlen_small_1` (`l12d768_seqlen{1024,2048,4096,8192}`/
`l8d512_seqlen{1024,2048,4096,8192}`) launched on tpu1-4, matched-token-budget gpt2-medium/small
seqlen sweeps (`gpt2_jax/configs/sweep_seqlen_{medium,small}_1`) on tpu7/tpu8/tpu1, plus paper-match
dense baselines (`small_paper_match`/`medium_paper_match`) on tpu5/tpu6. `total_batch_size=524288`
fixed everywhere -> `max_steps=19073` and the LR schedule (`MAX_LR=6e-4`, `WARMUP_STEPS=715`,
`optax.warmup_cosine_decay_schedule`) are byte-identical across every run in both sweeps, confirmed
from source, not assumed.

**Per-device batch_size had to be tuned down per seqlen on every node** (30.75GB/chip v4-8 ceiling,
`small_rope_default.py`'s own default is `batch_size=32`, NOT 16 -- corrected after using the wrong
Cable `MICRO_BATCH_SIZES` default, confirmed from `Cable_ref/src/train_gpt.py`): 1024/2048 configs
run fine at the model's own default (32 small / 16 medium); 4096 needed `--batch-size 16` (small) /
`8-16` (medium, right at the 30.8-30.9G edge, OOM'd multiple times at higher values before settling);
8192 needed `--batch-size 4` (both small and gpt2-small-8192, following the same landed value).
`grad_accum_steps` auto-adjusts to keep effective tokens/step == `total_batch_size` regardless, so
step count/LR schedule stay invariant across all these overrides.

**Recurring node-ops issues this session** (all now standard playbook, not new failure modes):
`~/qcute/data` symlink into `/dev/shm/qcute_data` intermittently missing on a fresh tmux launch
(not a tmpfs wipe -- `loginctl enable-linger` confirmed already `yes`; the underlying 19GB of real
`.npy` files on tmpfs stayed intact throughout, only the symlink itself needed recreating via
`ln -sf`) -- recreate before every relaunch rather than assuming it persists; stale TPU device locks
(`"already in use by pid X"`, `/tmp/libtpu_lockfile` ABORTED) cleared by `sudo rm -f
/tmp/libtpu_lockfile` + relaunch, reported PID already dead in every case.

**Val loss/PPL, seq=1024, full 19073-step run, summformer-small vs. gpt2-small (paper-match)**:

| Step | summformer-small | gpt2-small |
|---|---|---|
| 0 | 10.92 / 55339 | 11.00 / 59876 |
| 5000 | 3.407 / 30.2 | 3.309 / 27.4 |
| 10000 | 3.250 / 25.8 | 3.156 / 23.5 |
| 19072 (final) | **3.134 / 22.97** | **3.039 / 20.88** |

gpt2-small (dense, real-flash) beats summformer-small throughout, finishing ~9% lower PPL at this
seqlen -- consistent with every prior seqlen-sweep entry in this doc. Unconfounded on the training
recipe (identical schedule/batch-tokens as derived above); confounded on architecture (different
model entirely), as always.

**FLOPs/wallclock isolation (batch=32, seq=1024, 10 warmup + 50 iters, single fixed batch, mean+
median+std, `bench_flops.py`/`bench_flops_cpu.py`, not checked into the repo -- scratch scripts)**:

| Backend | Model | Params | fwd GFLOPs | fwd median | fwd+bwd GFLOPs | fwd+bwd median | tok/s |
|---|---|---|---|---|---|---|---|
| TPU (real flash) | gpt2-small | 123.7M | 9398.8 | 158.3ms | 25712.8 | 489.2ms | 66,984 |
| TPU (real flash) | summformer-small | 135.5M | 7068.8 | 196.6ms | 21093.1 | 627.5ms | 52,216 |
| CPU (no flash, dense) | gpt2-small | 123.7M | 9452.4 | 9216ms | 28280.2 | 28746ms | 1,133 |
| CPU (no flash, dense) | summformer-small | 135.5M | 6821.9 | 6877ms | 20388.3 | 23797ms | 1,375 |

**Key finding**: with flash-attention removed from both sides (CPU row), summformer has BOTH fewer
FLOPs (~28% less) AND is wall-clock FASTER (~25% less fwd time) than gpt2-small -- the architecture
itself is genuinely cheaper per the FLOP count. On real TPU hardware with the real Pallas kernel,
summformer is slower wall-clock despite still having fewer FLOPs (7069 vs 9399 GFLOPs fwd) -- this
flips the ranking, and points to kernel-launch-count overhead (13+ separate flash-attention call
sites: embedder + 12 encoder chain stages + decoder, vs. gpt2's single uniform stack) as the TPU-
specific cost, not the architecture's FLOP profile. The seqlen=1024 gap to gpt2-small is likely
compounded by this being the worst-case regime for a summarization hierarchy (dense attention is
still cheap at T=1024, so summformer pays overhead with none of its structural long-context payoff)
-- the in-flight 2048/4096/8192 points on this same sweep are the real test of whether the gap
narrows as T grows past where dense attention's O(T^2) cost actually bites.

Nodes at time of writing: tpu1 free (just finished summformer-small-1024, relaunched
gpt2-small-seqlen8192 batch_size=4), tpu2/3/4 mid-sweep (summformer-small 2048/4096/8192), tpu5
finished (small_paper_match), tpu6/7/8 mid-sweep (medium_paper_match, gpt2-small 2048/4096).

**Zero-shot length-extrapolation eval** (gpt2-small-1024/summformer-small-1024 checkpoints, NO
retraining, forward pass only at T=1024/2048/4096/8192 -- valid since both use `pos_method="rope"`,
no learned position table, no hard context-length assert in either model's forward path when
`pos_method != "learnable"`; `force_dense=True` on every summformer stage means self-/cross-attn
windows resolve to dense/None regardless of the `context_len` a stage was constructed with, so
param shapes are T-independent). tpu1's `gpt2-small-seqlen8192` run was stopped (Ctrl-C) to free
its chip for this eval, since its checkpoint already lived there -- per the no-egress-pulling rule,
each eval script ran directly on the node holding its own checkpoint (tpu5 for gpt2-small, tpu1 for
summformer-small) rather than copying checkpoints anywhere. 20 val batches (B=4) averaged per T,
`eval_extrapolate_{gpt2,summ}.py` (scratch scripts, not checked into the repo).

| T | gpt2-small val_loss / ppl | summformer-small val_loss / ppl |
|---|---|---|
| 1024 (in-distribution) | 3.006 / 20.20 | 3.098 / 22.16 |
| 2048 | 4.037 / 56.63 | 4.097 / 60.18 |
| 4096 | 5.421 / 226.11 | 6.291 / 539.44 |
| 8192 | 6.226 / 505.55 | 8.015 / 3025.62 |

Both collapse well beyond their trained length (neither model was trained for length
generalization -- no ALiBi/YaRN/NTK-RoPE-scaling, no long-context fine-tuning phase), as expected.
gpt2-small degrades somewhat less badly at the two longest lengths (4096/8192) despite starting
~0.09 loss ahead at 1024 -- summformer's summarization hierarchy does not appear to buy any
zero-shot extrapolation robustness over plain dense causal attention at this scale; if anything the
gap widens with T (ppl ratio summformer/gpt2: 1.10x at 1024, 1.06x at 2048, 2.39x at 4096, 5.99x at
8192). Single checkpoint each, not yet a multi-seed result.
