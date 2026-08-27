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
logic byte-for-byte the same, only cosmetic diffs). `summformer_jax/data_loader.py` vs.
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

**Current live status, all 4 nodes** (paper-faithful `total_batch_size=524288` throughout, checked
2026-08-27 ~08:23 UTC — steps/loss will be stale by the time this is read, tok/s is the stable
number):

| Node | Run | Step | Loss | tok/s | Notes |
|---|---|---|---|---|---|
| tpu4 | `medium_paper_match_b8` | 281 | 5.66 | ~102.6K | gpt2-medium baseline, batch_size=8/grad_accum=16 |
| tpu5 | `summformer_medium_ablation` | 336 | 5.6-5.7 | ~250.2K | ablation vs. tpu4 |
| tpu6 | `small_paper_match` | 362 | 5.2-5.5 | ~257.8K | gpt2-small baseline, batch_size=32/grad_accum=4 (needed a fresh `gpt2_jax` code sync — its `model_gpt.py` was stale/pre-bf16, caused one crash-and-relaunch) |
| tpu7 | `summformer_small_ablation` | 885 | 4.4-4.7 | ~599K | ablation vs. tpu6 |

Ablation runs are faster in raw tok/s than their baselines (smaller `n_layers` more than offsets
the fuse-stage FLOPs overhead at these sizes) — not itself meaningful, the actual comparison is
loss/bpb at matched token counts once both sides have logged enough steps. It's expected/fine for
the ablation to lose to its baseline if it finishes — the point is an honest comparison at
roughly matched compute, not tuning to win (explicit instruction).

ETAs at these steady-state speeds (19,073 steps total, 1 epoch): tpu4 ~27hr, tpu5 ~11hr, tpu6
~11hr, tpu7 ~5hr — all multi-hour, so don't stream every step; check in periodically instead (see
below).

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
