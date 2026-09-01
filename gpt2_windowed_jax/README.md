# gpt2_jax

JAX/Flax NNX port of [Cable](https://github.com/axiomlab/Cable)'s PyTorch nanoGPT
(`Cable/src/model_gpt.py`), restricted to 3 `pos_method`s (`rope`, `learnable`, `base`/NoPE),
trained on FineWeb-Edu 10B via Cable's own `dataset_preparation.py` (unmodified). Multi-device
training uses `jax.pmap` across all local TPU chips. Full background/design rationale, current
run status, and the running log of what's been tried: [../docs/status_tpu.md](../docs/status_tpu.md).
General TPU node setup (direct-ssh, tmux, disk-stall postmortem): [../docs/tpu_setup.md](../docs/tpu_setup.md).

This doc is the step-by-step for a fresh session picking this lineage back up on a TPU node.
**Never create a TPU yourself** — only use nodes already listed in [../TPU.md](../TPU.md).

## 1. Set up env (copy to node, run)

Set up the direct-ssh persistent connection first (see
[../docs/tpu_direct_ssh.md](../docs/tpu_direct_ssh.md)) — every command below assumes
`-o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p` is already usable.

```bash
# from local repo root, copy the repo (exclude venv/data/logs/checkpoints -- see tpu_setup.md)
rsync -avz -e "ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p" \
  --exclude='.venv' --exclude='.git' --exclude='datasets' --exclude='data' \
  --exclude='logs' --exclude='checkpoints' --exclude='__pycache__' --exclude='*.pyc' \
  ./ muaz@<external_ip>:~/qcute/

# on the node: uv sync now resolves jax[tpu]/flax/optax/orbax-checkpoint/tpu-info correctly on
# its own (jax-ai-stack's hard pin was dropped from pyproject.toml 2026-08-27) -- no manual
# `uv pip install -U jax[tpu]` follow-up needed anymore.
ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p muaz@<external_ip> \
  'cd ~/qcute && ~/.local/bin/uv sync'
```

`.env` (for `HF_TOKEN` etc.) still needs a separate `scp` — not part of the rsync exclude list
above, but also not committed, so copy it explicitly if the node doesn't already have one.

## 2. Prep data to RAM

`dataset_preparation.py` is Cable's own script, copied verbatim except it defaults
`HF_HOME`/`HF_DATASETS_CACHE` to `/dev/shm` (tmpfs) — **do not remove that**, writing HF
`datasets`' cache to persistent disk on these nodes has caused a confirmed multi-hour
uninterruptible-disk-sleep stall (see status_tpu.md's postmortem). Run from `~/qcute`:

```bash
python3 gpt2_jax/dataset_preparation.py --dataname fineweb-edu-10B
```

Writes shards to `~/qcute/data/fineweb-edu-10B/` (`train_*.npy`/`val_*.npy`, uint16 GPT-2 BPE
token ids). Safe to re-run/resume; takes a while (streams+tokenizes the whole 10B-token set).
Run this inside `tmux` too — it's long enough to be worth detaching from.

## 3. Run

Always launch inside a `tmux` session (never a bare blocking `ssh --command`) so it survives a
dropped connection and so you/the user can attach and watch it live. Prefer a `--config` file
(qcute-style, CLI flags override it) over hand-written flags — `configs/gpt2_jax/medium_rope_default.py`
and `small_rope_default.py` are the paper-faithful configs (see the formula below):

```bash
tmux new-session -d -s <run_name> "\
  export LD_LIBRARY_PATH=/home/muaz/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/lib:\$LD_LIBRARY_PATH && \
  cd ~/qcute && source .venv/bin/activate && \
  python3 gpt2_jax/train_gpt.py --config configs/gpt2_jax/medium_rope_default.py --run-name <run_name> \
    > ~/<run_name>.log 2>&1; echo TRAIN_EXIT=\$? >> ~/<run_name>.log"
```

- Watch it live: `tail -f ~/<run_name>.log` (raw stdout/stderr) and/or
  `tail -f logs/<run_name>/log.jsonl` (structured, same data) on the node; `tmux attach -t
  <run_name>` or `tmux capture-pane -t <run_name> -p -S -N` (peek without attaching) from another
  session.

### Formula to match Cable's paper baseline (any model size, any device count)

Cable trained on 8x H100; this port runs on 4-chip TPU v4-8 nodes. To stay faithful to the paper
despite the different device count, keep the paper's own per-device `batch_size` (`MICRO_BATCH_SIZES`
in `train_gpt.py`: 64/32/16 for tiny/small/medium) AND its own `total_batch_size=524288` fixed --
let `grad_accum_steps` absorb the device-count difference, exactly per the paper's own formula:

```
grad_accum_steps = total_batch_size // (batch_size * sequence_length * n_devices)
```

This is what `configs/gpt2_jax/{small,medium}_rope_default.py` set explicitly (not left to CLI
defaults, so the recipe is self-documenting). **Caveat found 2026-08-27**: `medium`'s own
batch_size=16 at this total_batch_size (grad_accum=8, which forces `train_gpt.py`'s fused
`jax.lax.scan` grad-accum path) OOMs by a small margin (~340MB) even with flash-attention+bf16 --
more headroom is needed for the scanned path than the grad_accum=1 case ever needed. Fix: pass
`--batch-size 8` (grad_accum=16) for medium at this total_batch_size; small's own batch_size=32
(grad_accum=4) fits without adjustment. Both still yield the SAME per-step token count the paper's
own grad_accum=4(small)/2(medium) on 8 GPUs would (only more microbatches per step, not a
different sequence of gradient updates).

### Baselines vs. `summformer_jax`'s ablation runs (2026-08-27)

`summformer_jax/` (see [../summformer_jax/README.md](../summformer_jax/README.md)) ablates the
Ks-hierarchical-summarization + fuse-cross-attn method against these exact baselines, at matched
`total_batch_size`, matched d_model/n_heads, and the confirmed-identical (see status_tpu.md's
dataloader audit) `DataLoaderLite`/dataset. Status as of the last check:

| Node | Run | Config | Status |
|---|---|---|---|
| tpu4 | `medium_paper_match_b8` | gpt2-medium baseline, `batch_size=8`, paper `total_batch_size=524288` | running, step 6922, ~102.6K tok/s |
| tpu5 | `summformer_medium_ablation` | summformer ablation vs. medium (`Ks=(2,2,2)`, `n_layers=2`, `d_model=1024`) | running, step 17838, ~293K tok/s |
| tpu6 | `small_paper_match` | gpt2-small baseline, `batch_size=32`, paper `total_batch_size=524288` | running, step 16949, ~258K tok/s |
| tpu7 | `summformer_small_ablation` | summformer ablation vs. small (`Ks=(2,2,2)`, `n_layers=1`, `d_model=768`) | **finished** (19072/19073), val loss 3.20 / bpb 4.62, checkpoint 448M — node now idle, used for dataset-prep smoke tests |

Param counts: medium baseline 353.8M vs. ablation 279.3M (-21%); small baseline ~123.6M vs.
ablation ~126.8M (+2.6%). It's expected/fine for the ablation to lose to its baseline if it
finishes -- the point is an honest comparison at roughly matched compute, not tuning to win.

## 4. Monitor tpu-info

`tpu-info` is a tracked dependency (`pyproject.toml`) — no ad-hoc install needed, `uv sync`
already put it in `.venv`:

```bash
export LD_LIBRARY_PATH=/home/muaz/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/lib
cd ~/qcute && .venv/bin/tpu-info
```

Reports per-chip HBM usage and duty cycle (TensorCore Utilization/latency panels are N/A on
these nodes — not a bug, just unsupported metrics here). A duty cycle well under 100% with
memory not near the ceiling is the signal to look at host/XLA dispatch overhead rather than
memory tuning — see status_tpu.md's MFU discussion for what's already been tried.

**Reference numbers** (tpu4, `medium`, batch=16/grad_accum=1/bf16/flash-attn, 2026-08-27): the
exact command that produced this (`medium_opt_v1`, adds the vmem `LIBTPU_INIT_ARGS` flag on top
of the section-3 template above):

```bash
tmux new-session -d -s medium_opt_v1 "\
  export LD_LIBRARY_PATH=/home/muaz/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/lib:\$LD_LIBRARY_PATH && \
  export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304' && \
  cd ~/qcute && source .venv/bin/activate && \
  python3 gpt2_jax/train_gpt.py --model medium --pos-method rope \
    --dataset-dir data/fineweb-edu-10B --use-flash-attention \
    --batch-size 16 --total-batch-size 65536 --run-name medium_opt_v1 \
    > ~/medium_opt_v1.log 2>&1; echo TRAIN_EXIT=\$? >> ~/medium_opt_v1.log"
```

hits **~94.6K tokens/s** steady-state, **100% duty cycle** on all 4 chips, MFU ≈21%. The
`LIBTPU_INIT_ARGS` vmem flag itself hasn't been ablated separately yet (see status_tpu.md) — the
throughput jump is attributed mainly to the code fix below, not confirmed isolated from the flag.
That's after fusing the grad-accum/eval microbatch loops into a
single `jax.lax.scan`+`pmap` call each (the old per-microbatch Python loop called `float()` on a
device array every iteration, a blocking host sync that left duty cycle at ~72-87% despite bf16
already being on) plus a background data-prefetch thread and `donate_argnums` on the optimizer
apply step — all baked into `train_gpt.py` now, not an opt-in flag. If a run comes in noticeably
under ~90K tok/s or duty cycle well under 100%, something regressed — check `git log
gpt2_jax/train_gpt.py` against this baseline before re-deriving fixes from scratch. Full
progression/numbers for every intermediate config tried today: status_tpu.md.

## Warning: do not pull checkpoints — up to the user, due to egress

**Do not `scp` checkpoints off a TPU node on your own judgement.** They're large binary files;
pulling one costs real egress and is the user's call, not a default action. This mirrors the
existing standing rule for this repo (already in
[../CLAUDE.md](../CLAUDE.md) and [../docs/tpu_setup.md](../docs/tpu_setup.md) for the other TPU
lineages) — same policy applies here, not a new one.

**Do pull `logs/<run_name>/log.jsonl` periodically** (roughly hourly for a multi-hour run) — it's
small structured JSON, cheap on egress, and is what you'd plot/compare across runs. Same command
also grabs `resolved_config.py` (the fully-resolved config snapshot, a few KB, useful to keep
alongside the log for reproducibility) — both live in the same `logs/<run_name>/` dir, neither is
a "big artifact" like a checkpoint:

```bash
mkdir -p logs/<run_name>  # first pull only
scp -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p \
  muaz@<external_ip>:~/qcute/logs/<run_name>/log.jsonl \
  muaz@<external_ip>:~/qcute/logs/<run_name>/resolved_config.py \
  logs/<run_name>/
```

Do not pull the raw `~/<run_name>.log` (tqdm/stderr noise, redundant with `log.jsonl`, larger) or
anything under `logs/<run_name>/model_*` (orbax checkpoint dirs — same egress rule as above).
