# Setting up `qcute.bytelm_tpu` on a TPU VM

End-to-end steps to get from a bare TPU VM (see [TPU.md](../TPU.md) for available queued
resources — **never create a new one yourself**, only use nodes already listed there) to a
running `qcute.bytelm_tpu` training job. Session-tested on a `v6e-1` node
(`v2-alpha-tpuv6e` runtime, Ubuntu 22.04). **Right after step 0 below (confirming the node is
READY), set up direct ssh per [docs/tpu_direct_ssh.md](tpu_direct_ssh.md) before doing anything
else** — every step from 1 onward should go through that persistent connection, not repeated
`gcloud ... ssh` calls; this doc's own command blocks are still written with plain
`gcloud ... ssh` for copy-paste clarity, substitute the direct-ssh form once the connection is
live.

## Run long/interactive work inside `tmux` on the VM, not a bare `gcloud ssh --command`

Any command that takes more than a few seconds, or that the user should be able to watch/attach
to live — installs, data prep, and especially training — should run inside a `tmux` session on
the TPU VM itself, not as a single blocking `gcloud ... ssh --command="..."` call. A bare
`--command` call ties up the local shell for the whole duration and gives the user no way to
reattach if they want to look in later; `tmux` decouples the remote process's lifetime from any
one SSH connection and lets the user (or a later Claude session) reattach at any time. `tmux` is
preinstalled on the standard TPU VM image (Ubuntu 22.04, confirmed `tmux 3.2a`).

Launch inside a named session:

```bash
gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> \
  --command="tmux new-session -d -s bytelm 'cd ~/qcute && <the actual long-running command>; exec bash'"
```

(the trailing `; exec bash` keeps the pane open after the command exits/crashes, instead of the
session vanishing — useful for post-mortem.) Give the user the exact attach command so they can
watch it themselves:

```bash
gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> -- -t "tmux attach -t bytelm"
```

(`Ctrl-b d` detaches without killing the session.) To check on it without fully attaching:

```bash
gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> \
  --command="tmux capture-pane -t bytelm -p -S -40"   # last 40 lines
```

## 0. Get the node name + confirm it's READY

The queued-resource name (e.g. `tpu1r`) and the actual node name (e.g. `tpu1`) can differ —
gcloud ssh's own `Finished preparing node <name>.` line reveals the real one.

```bash
gcloud compute tpus queued-resources describe <qr-name> --project raden-tpu --zone <zone> \
  --format="value(state.state)"
# must print READY (or ACTIVE for the queued resource itself)
```

## 0.5. Set up direct ssh — do this now, before anything else

See [docs/tpu_direct_ssh.md](tpu_direct_ssh.md) in full. Short version: one `gcloud ... ssh
--command="echo ok"` call (propagates your key), get the node's external IP via
`gcloud compute tpus tpu-vm describe <node-name> --format="yaml(networkEndpoints,state)"`, then

```bash
mkdir -p ~/.ssh/controlmasters
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ControlMaster=auto \
  -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -o ControlPersist=6h \
  -i ~/.ssh/google_compute_engine -fN muaz@<external_ip>
```

From here on, every command on this node uses
`ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@<external_ip> "<command>"`
(and `scp` with the same `-o ControlPath=...`) instead of `gcloud ... ssh`/`scp` — this doc's
remaining command blocks show the `gcloud` form for copy-paste clarity only.

## 1. Install `uv` on the VM

The stock VM ships Python 3.10 and no `uv`; `uv` will fetch its own Python 3.12 (this repo's
`pyproject.toml` requires `>=3.12`).

```bash
gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> \
  --command="curl -LsSf https://astral.sh/uv/install.sh | sh"
```

## 2. Create the venv and install torch + torch_xla + deps

**Pin `torch` to exactly match `torch_xla`'s version** — letting the resolver pick torch's
latest (2.13.0 as of this session) against `torch_xla[tpu]`'s latest available (2.9.0) installs
fine but fails at import time (`undefined symbol` in `_XLAC...so`, an ABI mismatch). `torch_xla`
lags `torch`'s own release cadence, so always check what `torch_xla[tpu]`'s top available
version actually is and pin `torch==` to the same number.

```bash
gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> --command="
  source \$HOME/.local/bin/env && \
  cd ~ && mkdir -p qcute && cd qcute && \
  uv venv --python 3.12 && source .venv/bin/activate && \
  uv pip install 'torch==2.9.0' 'torch_xla[tpu]==2.9.0' \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    -f https://storage.googleapis.com/libtpu-wheels/index.html && \
  uv pip install matplotlib sentencepiece tqdm
"
```

(If a fresh install without the explicit pin ends up with mismatched versions, reinstall both
pinned to torch_xla's actual max — check with
`uv pip install 'torch_xla[tpu]' --dry-run -f https://storage.googleapis.com/libtpu-releases/index.html` first.)

### `libpython` not found

uv's standalone Python build doesn't put its shared library on the default loader path —
`import torch_xla` fails with `ImportError: libpython3.12.so.1.0: cannot open shared object
file`. Fix by adding it to `LD_LIBRARY_PATH` (needed on every subsequent Python invocation, not
just once):

```bash
export LD_LIBRARY_PATH=/home/muaz/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH
```

(path will differ if uv picks a different exact patch version — check with
`find ~/.local/share/uv/python -name 'libpython3.12.so.1.0'` if the above doesn't exist.)

## 3. Sanity-check torch_xla sees the TPU chip

```bash
gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> --command="
  cd ~/qcute && source .venv/bin/activate && \
  export LD_LIBRARY_PATH=/home/muaz/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/lib:\$LD_LIBRARY_PATH && \
  python3 -c \"import torch_xla.core.xla_model as xm; d=xm.xla_device(); print(d); import torch; x=torch.randn(4,4).to(d); print((x@x).sum())\"
"
```

Expect `WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.` then `xla:0` and
a real tensor sum. If this hangs or errors, stop — fix it before scp'ing the project (the most
likely causes are the torch/torch_xla version mismatch above, or the TPU having been preempted;
check `gcloud compute tpus queued-resources describe <qr-name> ... --format="value(state.state)"`).

## 4. scp the project over

Exclude `.venv`, `.git`, `datasets`, `logs`, `checkpoints`, `__pycache__` — none of those should
cross the wire (venv gets rebuilt remotely, datasets get regenerated, logs/checkpoints are
per-machine run state).

```bash
tar czf /tmp/qcute_src.tar.gz \
  --exclude='.venv' --exclude='.git' --exclude='datasets' --exclude='logs' \
  --exclude='checkpoints' --exclude='__pycache__' --exclude='*.pyc' \
  qcute configs scripts docs pyproject.toml uv.lock CLAUDE.md TPU.md

gcloud compute tpus queued-resources scp /tmp/qcute_src.tar.gz <qr-name>:~/qcute_src.tar.gz \
  --project raden-tpu --zone <zone>

gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> \
  --command="cd ~/qcute && tar xzf ~/qcute_src.tar.gz && rm ~/qcute_src.tar.gz && ls"
```

(`tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'`
warnings from a macOS-built tarball are harmless — ignore them.)

To re-sync after local edits, redo the tar+scp+extract (a plain `--exclude`d re-tar is cheap;
this repo has no rsync-over-gcloud-ssh path set up).

## 5. Prepare the dataset

```bash
gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> --command="
  source \$HOME/.local/bin/env && cd ~/qcute && source .venv/bin/activate && \
  python3 scripts/prepare_data.py
"
```

Downloads/writes `datasets/enwik8.gz` (full 100,000,000-byte corpus) and
`datasets/enwik8_1M.gz` (dev slice).

## 6. Smoke test: `--device xla` actually needed

`qcute.bytelm_tpu` (unlike plain `qcute.bytelm`/`qcute.qcute_v1.qcute_v1`) auto-detects and uses
the TPU by default — but always check the run's own startup log line for `device=xla(...)`, not
just that the process didn't crash; a silent CPU fallback (e.g. torch_xla import failing) is easy
to miss otherwise.

Launch inside `tmux` (see above) so the user can attach and watch it live:

```bash
gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> --command="
  tmux new-session -d -s bytelm 'cd ~/qcute && source \$HOME/.local/bin/env && source .venv/bin/activate && \
    export LD_LIBRARY_PATH=/home/muaz/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/lib:\$LD_LIBRARY_PATH && \
    python3 -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_overfit10k.py --device xla; exec bash'
"
```

Give the user the attach command:

```bash
gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> -- -t "tmux attach -t bytelm"
```

or peek without attaching:

```bash
gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> \
  --command="tmux capture-pane -t bytelm -p -S -40"
```

Check for:
- the startup line: `preset=xs  params=3.4M  device=xla:0  xla=True  ...`
- `val_bpb` printed every `eval_every` steps and trending down
- no traceback

Once that's confirmed good, kill/replace the session and launch the real target run the same way
(a fresh session name, e.g. `bytelm_sd`, avoids clobbering the smoke test's pane history):

```bash
gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> --command="
  tmux new-session -d -s bytelm_sd 'cd ~/qcute && source \$HOME/.local/bin/env && source .venv/bin/activate && \
    export LD_LIBRARY_PATH=/home/muaz/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/lib:\$LD_LIBRARY_PATH && \
    python3 -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_sd_full_enwik8.py --device xla; exec bash'
"
```

Attach with `tmux attach -t bytelm_sd` (same `-- -t "tmux attach -t bytelm_sd"` pattern above), or
watch `~/qcute/logs/bytelm_tpu_sd_full_enwik8/run.log` directly (structured, real-time — see
[CLAUDE.md](../CLAUDE.md)'s logging convention) — and per that same doc, **watch actual
elapsed-time/it-rate early** rather than trusting the config docstring's step estimate; retune
`--steps` (or edit the config and relaunch) once real throughput is known.

## Monitoring a multi-hour run

For a run sized in hours (not minutes), check in periodically — roughly hourly is a reasonable
default cadence, tighter early on if the step budget/throughput estimate hasn't been validated
yet — rather than leaving it fully unattended or babysitting continuously. Each check:

```bash
ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine \
  muaz@<external_ip> "tmux capture-pane -t bytelm -p -S -20"
```

note the latest `val_bpb` (and whether it's still falling or has plateaued/started climbing —
overfitting on a small corpus over many epochs is expected for small models, see
configs/bytelm_tpu/bytelm_tpu_sm_full_enwik8.py's own docstring), and confirm no traceback.

**Pull back only `run.jsonl`, not `run.log` or checkpoints, to save egress**: `run.jsonl` is the
small structured record `scripts/plot_run.py` reads; `run.log` is the same data plus tqdm
progress-bar noise (redundant, larger), and checkpoints are large binary files with no reason to
leave the TPU VM mid-run. Copy it to the *same relative path* under the local repo's own `logs/`
(so `scripts/plot_run.py logs/<run_name>` works locally without edits):

```bash
scp -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine \
  muaz@<external_ip>:~/qcute/logs/<run_name>/run.jsonl logs/<run_name>/run.jsonl
```

(`mkdir -p logs/<run_name>` first if this is the first pull for that run.) Re-running the same
command later just overwrites the local copy with the latest one — safe to repeat every check-in.

## Common failure modes recap

| Symptom | Cause | Fix |
|---|---|---|
| `uv: command not found` on a later ssh call | each `ssh --command=...` is a fresh non-login shell | `source $HOME/.local/bin/env` at the start of every command |
| `ImportError: libpython3.12.so.1.0` | uv Python's shared lib not on loader path | `export LD_LIBRARY_PATH=...` (step 2) |
| `undefined symbol: ...deleteNodeEPN...` importing `_XLAC` | torch/torch_xla version mismatch | pin both to torch_xla's exact max version (step 2) |
| `ssh ... exited with return code 255` / hangs | TPU preempted (spot), or transient SSH flakiness | check `queued-resources describe ... state.state`; retry |
| training log shows `device=cpu` | forgot `--device xla`, or torch_xla import silently failed | always check the startup log line, not just exit code |
