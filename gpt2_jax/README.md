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
dropped connection and so you/the user can attach and watch it live.

```bash
tmux new-session -d -s <run_name> "\
  export LD_LIBRARY_PATH=/home/muaz/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/lib:\$LD_LIBRARY_PATH && \
  cd ~/qcute && source .venv/bin/activate && \
  python3 gpt2_jax/train_gpt.py --model medium --pos-method rope \
    --dataset-dir data/fineweb-edu-10B --use-flash-attention \
    --batch-size 16 --total-batch-size 65536 --run-name <run_name> \
    > ~/<run_name>.log 2>&1; echo TRAIN_EXIT=\$? >> ~/<run_name>.log"
```

- `--config configs/gpt2_jax/*.py` also works in place of individual flags (qcute-style config
  file, CLI flags override it) — see that dir for existing configs.
- Per-device `--batch-size`/`--total-batch-size` are hardware-dependent, not universal — a v4-8's
  30.75GB/chip needs `--use-flash-attention` plus bf16 (on by default in `model_gpt.py`'s
  `ModelConfig`) to fit Cable's own batch_size=16 default; see status_tpu.md for the OOM history
  if retuning these for a different node/model size.
- Watch it live: `tail -f ~/<run_name>.log` (raw stdout/stderr) and/or
  `tail -f logs/<run_name>/log.jsonl` (structured, same data) on the node; `tmux attach -t
  <run_name>` or `tmux capture-pane -t <run_name> -p -S -N` (peek without attaching) from another
  session.

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
