# CLAUDE.md

Guidance to Claude Code for this repository.

## Commands

```bash
uv sync                                         # install/update env from pyproject.toml + uv.lock
uv run python scripts/prepare_data.py           # download/cut datasets/enwik8{,_1M}.gz
uv run python scripts/train_bpe.py --data datasets/enwik8_1M.gz   # BPE tokenizer for qcute.bpelm
uv run python -m qcute.bytelm --preset sd       # byte-level baseline LM (Phase 0), reports BPB
uv run python -m qcute.bpelm --sp_model datasets/bpe_enwik8_1M_8192.model   # BPE baseline
uv run python -m qcute.bytelm --config configs/bytelm_xs_mtp4_ctx1024.py   # named, reproducible byte-level run
uv run python scripts/plot_run.py logs/<run_name>   # train/val bpb PNG from a run's run.jsonl
uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks21_v256_pq1_overfit10k.py
uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks21_overfit10k.py
uv run python -m qcute.summformer.summformer --config configs/summformer/ks21_overfit10k.py
uv run python gpt2_jax/train_gpt.py --config configs/gpt2_jax/medium_rope_default.py
```

- `qcute_lagcodec` = ACTIVE latent-AR / parallel-block-local-decode lineage. Design doc: [docs/qcute_lagcodec_plan.md](docs/qcute_lagcodec_plan.md).
- `qcute_zero` = ACTIVE monolithic single-shared-LM lineage. See Architecture below.
- `summformer` = ACTIVE summary-token fusion transformer lineage. See Architecture below.
- `gpt2_jax` = ACTIVE TPU lineage (JAX/Flax port of Cable paper's nanoGPT, `Cable/src/model_gpt.py`, restricted to 3 `pos_methods`: rope/learnable/base-NoPE, FineWeb-Edu 10B via Cable's own `dataset_preparation.py`). See [docs/status_tpu.md](docs/status_tpu.md) for current run state.
- All modules read `--help` for full flags; support `--config path.py` (see `configs/` — each config's own docstring has its exact `uv run` invocation), `--run_name` (else derived from config/preset; logs/checkpoints key off it), `--eval_only --checkpoint_path ...`. `qcute.bytelm` also has `--qual_gen_bytes`.
- `xs` preset targets ~4 bytes/timestep (tiny-corpus scale) — see `qcute/bytelm.py`'s `PRESETS` comment.
- No test suite, linter, or CI yet.

## Training-run conventions

- **One training job at a time on local MPS.** Two concurrent processes contend for the GPU and both stall (observed directly). Kill/wait before launching another.
- **Never redirect stdout/stderr to `/dev/null`** when backgrounding a run — it swallows tracebacks and anything not routed through `Logger`. Use a file instead.
- **Never pipe through `tr '\r' '\n'`** to make tqdm readable — `tr` block-buffers its own stdout to a non-tty file, so `tail -f` sits empty for minutes even though the writer is flushing. Just redirect straight to a file (`... > /tmp/foo.log 2>&1 &`); `tail -f` shows new bytes in real time even though `cat` shows one long `\r`-blob.
- Use `pgrep -f "python3 -m qcute.<module>"` to find the training PID (`$!` after backgrounding gives the wrapper/shell PID, not Python's).
- **After launching, give the user**: the PID, a `tail -f` on the raw stdout/stderr file, a `tail -f` on `logs/<run_name>/run.log` (structured, flushed by `Logger` at `--log_every`/`--eval_every`), and — if in `tmux` — `tmux capture-pane -t <session> -p -S -N` plus `tmux attach -t <session>`, so they can watch live.
- Long runs have unpredictable throughput (observed: ~30min nominal budget taking 2.5-3.5h) — watch actual elapsed time/step rate early, don't assume schedule.
- **On a TPU node (always inside `tmux`), prefer no redirect at all** — launch the command directly (`tmux new-session -d -s <run_name> 'python3 ...'`) so tqdm/prints land straight in the pane; `tmux capture-pane`/`attach` then show live output with zero plumbing (confirmed 2026-08-29, `summformer_jax/image_gen/train.py`). Only redirect when the run's log needs periodic `scp`-pulling (monitoring routine below reads `.jsonl` off disk, not the pane) — in that case pipe through `tee`, not a plain `>`: `python3 ... 2>&1 | tee ~/<run_name>.log`. Confirmed `tee` doesn't have `tr`'s block-buffering bug (shows up in-file within ~2s). Plain redirect leaves the tmux pane blank by design; `tee` keeps it live too. This TPU/tmux guidance doesn't apply to local MPS (no pane to view there).

## TPU access

- **Fresh session doing a TPU run: read [docs/tpu_setup.md](docs/tpu_setup.md) and [docs/tpu_direct_ssh.md](docs/tpu_direct_ssh.md) first, in full.** [TPU.md](TPU.md) lists existing queued resources — never create a new one, never edit TPU.md itself (it's the user's own create-command list, not a session log).
- **Immediately after confirming a node is `READY`**, set up the direct-ssh persistent multiplexed connection (one `gcloud ... ssh` to propagate the key, then `ControlMaster`/`ControlPersist` against the external IP) and use that for every subsequent command — `gcloud ... ssh` re-validates/re-preps on every call (seconds of overhead each time). Full commands: [docs/tpu_direct_ssh.md](docs/tpu_direct_ssh.md).
- **Every TPU is a spot instance, can be preempted with no warning.** A previously-working node that suddenly can't be reached (hang, `Connection refused`, `No route to host`) → check `queued-resources describe ... state.state` for `PREEMPTED` before assuming a flaky connection or standing up a replacement (don't).
- **Recovering a dropped direct-ssh connection** (not preemption — a stale ControlMaster socket/expired host key): (1) `gcloud ... ssh --command="echo ok"` to re-propagate the key; (2) re-run the direct `ssh -o ControlMaster=auto ...` to rebuild the socket; (3) if `Host key verification failed` (IP reassigned to a different node), `ssh-keygen -R <external_ip>` first, then retry (2) with `-o StrictHostKeyChecking=accept-new`. Check `PREEMPTED` first regardless.
- **When launching/restarting any run, give the user in chat** (not just a doc) the exact `tmux attach -t <run_name>` and `tmux capture-pane -t <run_name> -p -S -N` commands. If the run redirects stdout to a log file, say so — the tmux pane stays blank by design and `tail -f` on the log is the real live view.
- **Never create/start a TPU yourself** — only use nodes already listed/running in TPU.md.
- Full scp-to-training walkthrough (uv/torch_xla install, failure modes, smoke test): [docs/tpu_setup.md](docs/tpu_setup.md). Long-running/monitorable remote commands go inside `tmux`, never a bare blocking `gcloud ... ssh --command`.
- **Multi-hour run monitoring**: check in ~hourly, pull back only `run.jsonl`/`log.jsonl` (not `run.log`/checkpoints) via `scp` to save egress. Report periodic pulls as **one combined table** (node, run, step, elapsed, train_bpb, val_bpb, test_bpb), not per-node prose; one-line note on anything crossing/approaching 1.0 bpb or otherwise notable (new best, run finished, crash). Silently check each node's last 4-5 evals for stall/divergence — only surface/act if it actually triggers (stop that node, report, ask before redesigning/relaunching — never unilaterally).
- **Default any TPU data-prep/dataloader cache to `/dev/shm` (tmpfs), not persistent disk.** Confirmed twice (2026-08-27, tpu4/tpu5, `gpt2_jax/dataset_preparation.py`): a large buffered write to persistent disk reliably drops the process into uninterruptible `D`-state once free disk drops under ~20GB — `kill -9` doesn't clear it, can take 30-40+ min on its own. Nodes have ~400GB RAM (~390GB free) vs. ~97GB disk, so point cache dirs at tmpfs (`HF_HOME`/`HF_DATASETS_CACHE=/dev/shm/...` etc.) by default. `gpt2_jax/dataset_preparation.py` already does this via `os.environ.setdefault(...)` before import — follow that pattern for new scripts. tmpfs doesn't survive reboot/preemption (acceptable — a preempted node needs relaunching anyway).
- **Before relying on `/dev/shm` long-lived on a fresh node, run `sudo loginctl enable-linger muaz` once.** Confirmed 2026-08-28 (tpu1/2/3): `systemd-logind`'s default `RemoveIPC=yes` wipes a user's tmpfs files once their last tracked login session ends — a `tmux new-session -d` over plain `ssh 'command'` does NOT keep the session alive from logind's view once that SSH call returns, even though the detached tmux/job keeps running. Symptom: silent `/dev/shm` data loss mid-run. Fix persists across reboots, do it right after direct-ssh setup: `ssh ... 'sudo loginctl enable-linger muaz'`. Doesn't affect persistent-disk paths.
- **torch_xla-specific findings** (flash-attention/nightly setup, `--multichip` hang, `zero_kv_sink` cost) apply only to the archived `qcute.bytelm_tpu` lineage — detail in [docs/tpu_setup.md](docs/tpu_setup.md). The active `gpt2_jax` lineage uses JAX, not torch_xla, and doesn't inherit that hang.
- **Current TPU run status**: [docs/status_tpu.md](docs/status_tpu.md) — living doc, check there rather than assuming anything here is current. Update in place (don't append) on every run start/stop/change.

## Architecture

Three enwik8 lineages are active (see Commands); current results/design state in [docs/status.md](docs/status.md) — this section only orients a fresh session to where code/docs live plus durable (non-dated) reference material.

- **`qcute_lagcodec`** (`qcute/qcute_lagcodec/`) — latent-AR / parallel-block-local-decode rewrite of `qcute_v5` (frozen at `qcute/v5_old/`): only the top level is a genuine NTP/AR decoder, every level below decodes via a per-block seed token. Design/plan: [docs/qcute_lagcodec_plan.md](docs/qcute_lagcodec_plan.md). `--decoder_type`: `stack` (default, non-interleaved, less memory), `stack_v1` (legacy interleaved-seed-token), `stack_local` (block-diagonal same-level conditioning), `stack_sync` (unimplemented stub) — see `qcute_lagcodec_decoder.py` docstrings.
- **`qcute_zero`** (`qcute/qcute_zero/`) — monolithic single-shared-LM: one LM does both the byte pass and every fuse-stage's code-sequence pass, periodic cross-attention back into the byte stream, no curriculum by design. Why this avoids `qcute_lagcodec`'s free-rollout collapse: [docs/archive5/status.md](docs/archive5/status.md). Formal bpb-validity writeup: [docs/maths.md](docs/maths.md). **Checkpoint caveat**: use `last.pt`, not `best.pt` — val_loss-based selection is a bad proxy until fixed.
- **`summformer`** (`qcute/summformer/`) — summary-token fusion transformer. Current design/results: [docs/status.md](docs/status.md).
- Every earlier fork (`qcutelm` family in `qcute/archive/`, `qcute_refine` family `v1.py`-`v4_5_1.py`/`qcute/archive2/`) is archived/superseded — one-line summary of every fork: [docs/archive/lineage_summary.md](docs/archive/lineage_summary.md). Original `qcutelm` design doc: [docs/archive/continuous_tokenizer_handover.md](docs/archive/continuous_tokenizer_handover.md). Pre-v4 KV-contribution probe narrative: [docs/archive2/kv_contribution.md](docs/archive2/kv_contribution.md).
- `qcute/bytelm.py` / `qcute/bpelm.py` are NOT archived — still the active baseline comparison points.
- **Standing methodology**: use a small (`n_bytes=10000`) slice with a short step budget (`configs/*_overfit10k_*.py`) as the fast-iteration testbed, until a config fast-overfits to a train bpb comparable to `qcute.bytelm`'s own parity numbers on the same slice (`configs/bytelm_overfit10k_*.py`). Full-scale runs/generation-quality comparisons aren't trustworthy before that bar is cleared.
- **Ks regression grid, simplest→hardest** (for config-writing): ranked by `product(Ks)` (compression ratio / min warm-up context) first, `n_levels` second, `max(Ks)` third. Ranks generation/architecture-correctness difficulty, not training difficulty.

| # | Ks | levels | product(Ks) | max K |
|---|---|---|---|---|
| 1 | `(1,)` | 1 | 1 | 1 |
| 2 | `(1,1)` | 2 | 1 | 1 |
| 3 | `(1,1,1)` | 3 | 1 | 1 |
| 4 | `(2,1)` | 2 | 2 | 2 |
| 5 | `(2,1,1)` | 3 | 2 | 2 |
| 6 | `(2,2)` | 2 | 4 | 2 |
| 7 | `(4,1)` | 2 | 4 | 4 |
| 8 | `(2,2,1)` | 3 | 4 | 2 |
| 9 | `(4,2)` | 2 | 8 | 4 |
| 10 | `(2,2,2)` | 3 | 8 | 2 |
| 11 | `(4,2,1)` | 3 | 8 | 4 |
| 12 | `(4,4,2)` | 3 | 32 | 4 |

## Code style

- Docstrings/comments extremely concise — code should be self-descriptive, comments ≤2 lines. Module-level (top-of-file) docstrings exempt but still kept tight.

## Response format

- Prefix every reply with the current timestamp (run `date` — never guess it).
- Keep chat replies terse: results and next steps directly, no restated context, no padding, no multi-paragraph recaps. Detail goes in `docs/status.md`, not chat.
- When comparing two runs/configs that differ in more than one variable, flag causal claims as "suspect"/"maybe" — an unconfounded isolation (one variable changed at a time) is required before stating cause as fact.
