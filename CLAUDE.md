# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
                                                 # ACTIVE lineage: qcute_lagcodec (qcute/qcute_lagcodec/) — the
                                                 # latent-AR / parallel-block-local-decode rewrite.
                                                 # Full design doc: docs/qcute_lagcodec_plan.md.
uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks21_overfit10k.py
                                                 # ACTIVE lineage: qcute_zero (qcute/qcute_zero/) — monolithic
                                                 # single-shared-LM design, see Architecture below.
uv run python -m qcute.summformer.summformer --config configs/summformer/ks21_overfit10k.py
                                                 # ACTIVE lineage: summformer (qcute/summformer/) — summary-token
                                                 # fusion transformer, see Architecture below.
uv run python gpt2_jax/train_gpt.py --config configs/gpt2_jax/medium_rope_default.py
                                                 # ACTIVE TPU lineage as of 2026-08-27: JAX/Flax port of the
                                                 # Cable paper's nanoGPT (Cable/src/model_gpt.py), restricted to
                                                 # 3 pos_methods (rope/learnable/base-NoPE), FineWeb-Edu 10B via
                                                 # Cable's own dataset_preparation.py used as-is. See
                                                 # docs/status_tpu.md for current run state.
```

`qcute.bytelm`, `qcute.bpelm`, `qcute.qcute_lagcodec.qcute_lagcodec`, `qcute.qcute_zero.qcute_zero`, and
`qcute.summformer.summformer` all read `--help` for their full flag
list; all support `--config path.py` (see `configs/` — every config file
has its own module docstring explaining what it's testing and its exact
`uv run` invocation, copy-pasteable directly from the file), `--run_name`
(else derived from `--config`/preset — logs and checkpoints both key off
it: `logs/<run_name>/`, `checkpoints/<run_name>/`), and `--eval_only
--checkpoint_path ...`; `qcute.bytelm` additionally supports
`--qual_gen_bytes` for qualitative generation. Tiny-corpus-scale defaults
(`xs` preset) target ~4 bytes/timestep — see `qcute/bytelm.py`'s
`PRESETS` comment for why. No test suite, linter, or CI config exists yet.

**Only ever run one training job at a time.** All four modules train on
MPS; two concurrent training processes contend for the same GPU and both
slow down (observed directly: a second run caused an already-progressing
job to stall with zero throughput). Kill or wait out the current run
before launching another — never launch a second training process while
one is still active.

**When launching a training run in the background, never redirect its
stdout/stderr to `/dev/null`** — use a file instead (e.g. a scratchpad
path, or `/tmp/<pid>.log` renamed once the PID is known post-launch),
since `/dev/null` silently swallows uncaught-exception tracebacks and
anything not routed through `Logger`, making crashes invisible. Do NOT
pipe through `tr '\r' '\n'` to make tqdm's `\r`-updates readable —
confirmed directly that `tr` itself full-block-buffers its own stdout
when writing to a non-tty file, so a `tail -f` on the piped-through file
sits empty for seconds/minutes at a time regardless of how eagerly the
Python process flushes its side of the pipe; it's not a live view, just a
deferred dump. Redirect stdout/stderr straight to a file instead (plain
`... > /tmp/foo.log 2>&1 &`, no pipe) — the tqdm line will look like one
long `\r`-joined blob when catted, but `tail -f` still shows new bytes
arriving in real time, which is the actual goal. Use `pgrep -f "python3
-m qcute.<module>"` to find the training process's PID (e.g. to kill it;
`$!` after a background launch gives the wrapper/shell PID, not
necessarily Python's). **After launching, give the user the PID, two
`tail -f` commands, and (when the run is in a `tmux` session, e.g. on a
remote TPU node) a `tmux capture-pane` command**: one `tail -f` on that
raw stdout/stderr file, one on `logs/<run_name>/run.log` (the structured
log `Logger` writes to at `--log_every`/`--eval_every` intervals,
genuinely real-time since `Logger` opens and flushes that file directly,
no pipe involved), and `tmux capture-pane -t <session> -p -S -N` (peeks
at the session's recent output without attaching — swap `-N` for how many
lines back) alongside the `tmux attach -t <session>` command — so they
can watch it live themselves rather than relying on being told the
outcome later. Long runs have shown unpredictable throughput (observed: a
nominal ~30-minute budget taking 2.5-3.5 hours instead) — watch actual
elapsed time/step rate early on rather than assuming a run will finish on
schedule.

**On a TPU node specifically, since the run always launches inside `tmux` (see TPU access below),
pipe through `tee` instead of a plain `>` redirect**: `python3 ... 2>&1 | tee ~/<run_name>.log`,
not `python3 ... > ~/<run_name>.log 2>&1`. Confirmed directly (2026-08-27) that GNU `tee` does NOT
have `tr`'s block-buffering problem — a `\r`-updating loop piped through `tee` showed up in the
file within ~2s, same as a direct redirect. The payoff: `tmux capture-pane` on that session then
shows real live output (matching what `tail -f` on the file shows), instead of an empty pane by
design the way a plain redirect leaves it. This is TPU/tmux-specific — the local MPS training
guidance above (background `&`, no tmux pane to view) is unaffected, plain redirect stays fine
there.

## TPU access

**Starting a fresh session asked to do a TPU run: read [docs/tpu_setup.md](docs/tpu_setup.md)
and [docs/tpu_direct_ssh.md](docs/tpu_direct_ssh.md) first, in full** — the first walks
scp → install → run → common failure modes (including two bugs already hit and fixed: the
torch/torch_xla version-pin ABI mismatch, and `xm.optimizer_step`'s `barrier=False` default
silently growing the XLA graph unboundedly across steps); the second is the direct-ssh
connection setup below, which is not optional — do it immediately, every time. Don't rediscover
any of this from scratch. [TPU.md](TPU.md) lists which queued resources exist (never create a
new one).

TPU VMs (see [TPU.md](TPU.md) for available queued resources) are reachable via
`gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone>`, but that
re-validates TPU state and re-preps the node on every call (several seconds of overhead each
time). **The very first thing to do on any fresh TPU node connection — right after confirming
it's `READY`, before install/scp/anything else — is set up the direct-ssh persistent multiplexed
connection** (one `gcloud ... ssh` call to propagate the key, then a `ControlMaster`/
`ControlPersist` session against the node's external IP) and use that for every subsequent
command on that node, not repeated `gcloud ... ssh` calls. **Every TPU listed in TPU.md is a
spot instance and can be preempted at any time with no warning** — if a node that was just
working suddenly can't be reached (hang, `Connection refused`, `No route to host`), check
`queued-resources describe ... state.state` for `PREEMPTED` *before* assuming a flaky connection,
retrying, or standing up a replacement node (don't — see above). Full setup and copy-pasteable
commands: [docs/tpu_direct_ssh.md](docs/tpu_direct_ssh.md).

**Recovering a dropped direct-ssh connection (confirmed 2026-08-27: 3 of 4 nodes' ControlMaster
sockets died simultaneously mid-session, `Permission denied (publickey)`/`Broken pipe` on direct
connect)** — not a preemption, just a stale multiplexed connection/expired host key. Fix, per
node: (1) `gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone>
--command="echo ok"` to re-propagate the SSH key (this alone re-validates and heals most cases);
(2) re-run the direct `ssh -o ControlMaster=auto -o ControlPersist=yes -o
ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@<external_ip>
'echo connected'` to rebuild the multiplexed socket; (3) if that fails with `Host key verification
failed` (the external IP was reassigned to a different node), `ssh-keygen -R <external_ip>` first,
then retry step 2 with `-o StrictHostKeyChecking=accept-new`. Check `queued-resources describe`
for `PREEMPTED` first regardless — this recovery path is for a healthy-but-unreachable node, not a
lost one.

**When launching or restarting any run on a TPU node, always give the user in chat (not only in a
doc) the exact `tmux attach -t <run_name>` and `tmux capture-pane -t <run_name> -p -S -N` commands
for that specific run** — same standing rule as above, applies to every lineage (`gpt2_jax`,
`summformer_jax`, not just the older torch ones). Note: if the launch script redirects stdout to a
log file (`> ~/<run_name>.log 2>&1`, the convention used by both JAX lineages), the tmux pane
itself stays blank by design — `tail -f ~/<run_name>.log` is the real live view in that case, still
worth giving alongside the tmux commands for peeking at the raw session.
**Never create/start a TPU yourself** — only use nodes already listed in TPU.md/already running.
**Never edit TPU.md itself** — it's the user's own list of queued-resource create commands, not a
session log; per-node details discovered while connecting (external IP, actual node name behind
a queued-resource name, accelerator type) go in a session's own working notes/commands, not
written back into TPU.md.
Full scp-to-running-training walkthrough (uv/torch_xla install, common failure modes, `qcute.
bytelm_tpu` smoke test): [docs/tpu_setup.md](docs/tpu_setup.md). **Any long-running
or user-monitorable remote command (installs, training) goes inside a `tmux` session on the TPU
VM**, not a bare blocking `gcloud ... ssh --command`, and the user gets the exact `tmux attach`
command back so they can watch it live themselves — see that doc's own tmux section for the
launch/attach/peek incantations. **For a multi-hour run, check in periodically (roughly hourly)
and pull back only `run.jsonl` (not `run.log` or checkpoints) to the matching local `logs/<run_name>/`
path to save egress** — see that doc's "Monitoring a multi-hour run" section for the exact
commands. **Report periodic pulls as one combined table** (columns: node, run, step, elapsed,
train_bpb, val_bpb, test_bpb — one row per active run across every node), not per-node prose —
keep the surrounding text to a one-line note on whether any run is approaching/crossed 1.0 bpb,
plus anything notable (new best, a run finished, a crash). Only fall back to fuller prose when
something needs explaining (e.g. the overfitting-response policy actually triggering).

**Default any TPU-node data prep/dataloader cache to RAM (`/dev/shm`, tmpfs), not persistent
disk.** Confirmed twice independently (2026-08-27, tpu4 and tpu5, `gpt2_jax/dataset_preparation.py`
via HF `datasets`): a large buffered write to the node's persistent disk — HF `datasets`' own
raw-parquet + re-encoded-Arrow cache, ~55-60GB, but the same shape of bug as the earlier
`prep_fineweb_edu_bytes.py`/`np.memmap` incident — reliably drops the writing process into an
uninterruptible `D` (disk-sleep) state with `/proc/<pid>/io`'s `write_bytes` frozen, once free
disk space drops under ~20GB. `kill -9` does **not** clear a `D`-state process; it can take 30-40+
minutes to clear on its own, there is no faster recovery. These nodes have ~400GB RAM with ~390GB
typically free (`free -h`), far more than the disk's 97GB — so point any data-prep cache dir at
tmpfs by default (`HF_HOME`/`HF_DATASETS_CACHE=/dev/shm/...` for HF `datasets`, or the equivalent
for whatever's being used) rather than waiting to hit this failure mode again. `gpt2_jax/
dataset_preparation.py` already does this (`os.environ.setdefault(...)` at import time, before
`datasets` is imported) — follow that pattern for any new TPU dataloader/prep script. tmpfs
content doesn't survive a node reboot/preemption, which is an acceptable tradeoff here (a
preempted node needs everything relaunched anyway).

**Before relying on `/dev/shm` for anything long-lived on a fresh TPU node, run `sudo loginctl
enable-linger muaz` once.** Confirmed 2026-08-28 (tpu1/2/3): `systemd-logind`'s `RemoveIPC=yes`
default wipes a user's tmpfs-owned files (including plain files under `/dev/shm`, not just SysV/
POSIX IPC objects) once their last tracked login session ends — and a `tmux new-session -d ...`
launched over a plain `ssh ... 'command'` does NOT keep a session alive from logind's point of
view once that SSH connection closes, even though the detached tmux process (and anything it
spawned) keeps running. Symptom: a multi-hour prep/training job silently loses its `/dev/shm` data
out from under it — once mid-training (~1h10-1h20 in, `FileNotFoundError` on a shard file, process
left hanging on a dead prefetch thread rather than crashing outright) and once immediately after a
prep script's own successful exit, before the SSH session had even fully wound down. Not a timer,
not a reboot, not disk pressure — `grep -i removeipc /etc/systemd/logind.conf` shows it's the
(commented-out, i.e. default) `RemoveIPC=yes`, and `loginctl show-user <user> --property=Linger`
showed `Linger=no`. Fix is one command, persists across reboots, do it immediately after the
direct-ssh setup on any node that will use `/dev/shm`: `ssh ... 'sudo loginctl enable-linger
muaz'`. Persistent-disk-based data (e.g. `gpt2_jax`'s own default output path) is unaffected by
this — it's specific to tmpfs.

**torch_xla-specific findings (flash-attention/nightly-build setup, the `--multichip` +
nightly-build hang investigation, `zero_kv_sink`+flash-attention's throughput cost) apply only to
the archived `qcute.bytelm_tpu` lineage** — full detail preserved in
[docs/tpu_setup.md](docs/tpu_setup.md), not repeated here. The active TPU lineage
(`gpt2_jax/`, see Commands above) uses JAX, not torch_xla, and doesn't inherit that hang (JAX
`pmap` across all local chips is expected to just work).

**Current TPU run status**: [docs/status_tpu.md](docs/status_tpu.md) — a living doc, check there
for what's running on tpu4/5/6/7 right now rather than assuming anything below is current. Update
that file in place (don't append) whenever a run starts/stops/changes.

**Standing monitoring routine**: pull each active run's `run.jsonl` via `scp` into local
`logs/<run_name>/` periodically (roughly hourly for a multi-hour run), print one combined table
(node, run, step, elapsed, primary metric), and silently check each node's runs for a stalled/
diverging metric over their last 4-5 evals — only surface/act on it if it actually triggers (stop
that node, report the sequence, ask before redesigning/relaunching — don't do it unilaterally).

## Architecture

Three enwik8 lineages are active (invocations under Commands above); current results/design state
for all three lives in [docs/status.md](docs/status.md), not here — this section only orients a
fresh session to where each lineage's code and design docs are, and gives durable (non-dated)
reference material.

- **`qcute_lagcodec`** (`qcute/qcute_lagcodec/`) — latent-AR / parallel-block-local-decode rewrite of
  `qcute_v5` (now frozen/archived at `qcute/v5_old/`): only the top level is a genuine NTP/AR
  decoder, every level below decodes via a per-block seed token (interleaved self-attention +
  cross-attention to that block's own code). Design narrative, worked examples, staged plan:
  [docs/qcute_lagcodec_plan.md](docs/qcute_lagcodec_plan.md). `--decoder_type`: `stack` (current default,
  non-interleaved, less memory), `stack_v1` (legacy interleaved-seed-token mechanism), `stack_local`
  (block-diagonal same-level conditioning), `stack_sync` (design-note stub, unimplemented) — see
  `qcute_lagcodec_decoder.py` docstrings for the difference between them.
- **`qcute_zero`** (`qcute/qcute_zero/`) — monolithic single-shared-LM design: one LM does both the
  byte pass and every fuse-stage's own code-sequence pass, periodic cross-attention back into the
  byte stream, no curriculum by design. Why this avoids `qcute_lagcodec`'s free-rollout collapse, its
  incremental KV cache, and full history: [docs/archive5/status.md](docs/archive5/status.md).
  Formal bpb-validity/paradigm-comparison writeup: [docs/maths.md](docs/maths.md). **Checkpoint
  caveat**: use `last.pt`, not `best.pt` — `Checkpointer`'s val_loss-based selection is a bad proxy
  for this lineage until fixed (that sum keeps climbing well past the point `val_byte_acc` is still
  improving).
- **`summformer`** (`qcute/summformer/`) — summary-token fusion transformer. Current design/results
  in [docs/status.md](docs/status.md).

Every earlier fork of both the `qcutelm` family (`qcute/archive/`, configs under
`configs/archive/`) and the `qcute_refine` family (`v1.py` through `v4_5_1.py`, `qcute/archive2/`)
is archived, superseded, kept only for historical reference — chronological one-line summary of
every fork in both:  [docs/archive/lineage_summary.md](docs/archive/lineage_summary.md). Original
design source-of-truth for the `qcutelm` lineage:
[docs/archive/continuous_tokenizer_handover.md](docs/archive/continuous_tokenizer_handover.md).
`qcute/bytelm.py` and `qcute/bpelm.py` are the exception — still the active baseline comparison
points, not archived. Pre-v4 `DecoderLevel` KV-contribution probe diagnostic
(`scripts/archive/probe_decoder_kv_contribution.py`) and its narrative:
[docs/archive2/kv_contribution.md](docs/archive2/kv_contribution.md).

**Standing methodology**: use a small (`n_bytes=10000`) slice with a short step budget as the
fast-iteration testbed for architecture changes — see `configs/*_overfit10k_*.py` — until a config
fast-overfits that slice to a train bpb comparable to `qcute.bytelm`'s own parity numbers on the
same slice (see `configs/bytelm_overfit10k_*.py`). Full-scale runs and generation-quality
comparisons aren't trustworthy before that bar is cleared.

**Ks regression grid, simplest to hardest** (for config-writing): ranked by `product(Ks)`
(compression ratio / minimum warm-up context) first, `n_levels` second, `max(Ks)` third. Ranks
generation/architecture-correctness difficulty (raggedness, warm-up depth), not
training/learnability difficulty.

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

Docstrings and comments must be extremely concise — assume code is self-descriptive; comments
never exceed 2 lines. Only the module-level (top-of-file) docstring is exempt from the length
limit, and should still be kept reasonably tight rather than accumulating restated history.

## Response format

Prefix every reply to the user with the current timestamp (run `date` for
the actual value — never guess it).

Keep chat replies terse. State results and next steps directly — no restating context the user
already has, no padding, no multi-paragraph recaps. Save detail for docs/status.md, not the chat.

When explaining a result from comparing two runs/configs that differ in more than one variable,
flag causal claims as "suspect"/"maybe" rather than stating them as established — an unconfounded
isolation (one variable changed at a time) is required before a cause can be stated as fact.
