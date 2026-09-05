# CLAUDE.md

Guidance to Claude Code for this repository.

## Commands

- `qcute_lagcodec` = ACTIVE latent-AR / parallel-block-local-decode lineage. Design doc: [docs/qcute_lagcodec_plan.md](docs/qcute_lagcodec_plan.md).
- `summformer` = ACTIVE summary-token fusion transformer lineage. See Architecture below.
- `image_gen_cifar` = ACTIVE CIFAR-10 hierarchical latent-AR image generator, single-file hard-fork of the `qcute_lagcodec` idea (`image_gen_cifar/run_causalattn.py`, no imports from `qcute_lagcodec`). Status/design: [docs/status_image_gen_cifar.md](docs/status_image_gen_cifar.md). `uv run python3 -m image_gen_cifar.run_causalattn --config image_gen_cifar/configs/base.py`; runs/checkpoints/configs land in `image_gen_cifar/logs/<run_name>/`, not the repo-root `logs/`.
- All modules read `--help` for full flags; support `--config path.py` (see `configs/` — each config's own docstring has its exact `uv run` invocation), `--run_name` (else derived from config/preset; logs/checkpoints key off it), `--eval_only --checkpoint_path ...`. `qcute.bytelm` also has `--qual_gen_bytes`.
- No test suite, linter, or CI yet.

## Training-run conventions

- **One training job at a time on local MPS.** Two concurrent processes contend for the GPU and both stall (observed directly). Kill/wait before launching another.
- **Never redirect stdout/stderr to `/dev/null`** when backgrounding a run — it swallows tracebacks and anything not routed through `Logger`. Use a file instead.
- **Never pipe through `tr '\r' '\n'`** to make tqdm readable — `tr` block-buffers its own stdout to a non-tty file, so `tail -f` sits empty for minutes even though the writer is flushing. Just redirect straight to a file (`... > /tmp/foo.log 2>&1 &`); `tail -f` shows new bytes in real time even though `cat` shows one long `\r`-blob.
- Use `pgrep -f "python3 -m qcute.<module>"` to find the training PID (`$!` after backgrounding gives the wrapper/shell PID, not Python's).
- **After launching, give the user**: the PID, a `tail -f` on the raw stdout/stderr file, a `tail -f` on `logs/<run_name>/run.log` (structured, flushed by `Logger` at `--log_every`/`--eval_every`), and — if in `tmux` — `tmux capture-pane -t <session> -p -S -N` plus `tmux attach -t <session>`, so they can watch live.
- Long runs have unpredictable throughput (observed: ~30min nominal budget taking 2.5-3.5h) — watch actual elapsed time/step rate early, don't assume schedule.
- **On a TPU node (always inside `tmux`), prefer no redirect at all** — launch the command directly (`tmux new-session -d -s <run_name> 'python3 ...'`) so tqdm/prints land straight in the pane; `tmux capture-pane`/`attach` then show live output with zero plumbing (confirmed 2026-08-29, `summformer_jax/image_gen/train.py`). Only redirect when the run's log needs periodic `scp`-pulling (monitoring routine below reads `.jsonl` off disk, not the pane) — in that case pipe through `tee`, not a plain `>`: `python3 ... 2>&1 | tee ~/<run_name>.log`. Confirmed `tee` doesn't have `tr`'s block-buffering bug (shows up in-file within ~2s). Plain redirect leaves the tmux pane blank by design; `tee` keeps it live too. This TPU/tmux guidance doesn't apply to local MPS (no pane to view there).
- **Always append `; exec bash` to a tmux launch command** — `tmux new-session -d -s <run_name> 'python3 ...; exec bash'`. Without it, the tmux session dies the moment the launched command exits, whether that's a crash, normal completion, or someone attaching and hitting Ctrl-C — the pane's shell has nothing left to run once the single command finishes, so the whole session vanishes along with any scrollback/output history. `exec bash` keeps the pane alive as an interactive shell afterward, so `tmux capture-pane`/`attach` still work for post-mortem inspection and Ctrl-C just interrupts the foreground process, not the session.

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
- **Current TPU run status**: [docs/status_tpu.md](docs/status_tpu.md) — living doc, check there rather than assuming anything here is current. Update in place (don't append) on every run start/stop/change.
## Code style

- Docstrings/comments extremely concise — code should be self-descriptive, comments ≤2 lines. Module-level (top-of-file) docstrings exempt but still kept tight.

## Response format

- Prefix every reply with the current timestamp (run `date` — never guess it).
- Keep chat replies terse: results and next steps directly, no restated context, no padding, no multi-paragraph recaps. Detail goes in `docs/status.md`, not chat.
- When comparing two runs/configs that differ in more than one variable, flag causal claims as "suspect"/"maybe" — an unconfounded isolation (one variable changed at a time) is required before stating cause as fact.
