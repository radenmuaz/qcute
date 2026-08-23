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

## Optional: nightly build, for `--use_flash_attention`

The stable pin above (torch/torch_xla 2.9.0) can't use `qcute.bytelm_tpu`'s
`--use_flash_attention` flag — that kernel needs `libtpu>=0.0.44`, and 2.9.0 locks `libtpu==0.0.21`
as its own dependency. Bumping libtpu alone on top of the stable pin is a confirmed hard break
(`RuntimeError: Unexpected PJRT_ExecuteOptions size: expected 112, got 80` — plugin/framework PJRT
API versions disagree). What works instead, confirmed directly on a `v4-8` node (Ubuntu 22.04):

```bash
uv venv --python 3.12 && source .venv/bin/activate
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-2.10.0.dev-cp312-cp312-linux_x86_64.whl \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.10.0.dev-cp312-cp312-linux_x86_64.whl
uv pip install libtpu jax   # left unpinned — resolves to libtpu 0.0.46 + matching jax/jaxlib
sudo apt-get update -qq && sudo apt-get install -y -qq libopenblas0   # nightly torch's wheel needs this; stable doesn't
```

(`UV_SKIP_WHEEL_FILENAME_CHECK=1` is needed because uv strictly checks wheel-filename-vs-metadata
version agreement, and this particular nightly wheel's internal metadata says `2.10.0+git...`
while the filename says `2.10.0.dev` — a real nightly-build quirk, not file corruption.) Verify
with the same device + tensor-op smoke test as step 3 above, then confirm flash-attention itself:

```bash
python3 -c "
import torch, torch.nn.functional as F, torch_xla
from torch_xla.experimental.custom_kernel import flash_attention
device = torch_xla.device()
q = torch.randn(8, 4, 4096, 64, device=device)
k = torch.randn(8, 4, 4096, 64, device=device)
v = torch.randn(8, 4, 4096, 64, device=device)
y_flash = flash_attention(q, k, v, causal=True, sm_scale=1.0/8.0)
y_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
torch_xla.sync()
print('max abs diff:', (y_flash - y_ref).abs().max().item())  # ~0.01 is normal (algorithm variance, not a bug)
"
```

Nightly builds are less stable than the pinned release — re-verify this whole chain (including
the generation-consistency check, `qcute.bytelm_tpu.validate_generation`) after any nightly
version bump, don't assume it still works.

## Optional: multiple TPU chips on one host

A multi-chip TPU slice (e.g. `v4-8` = 4 chips) exposes more than one device on a single VM.

**`qcute.bytelm_tpu --multichip` (true collective data-parallel training via `torch_xla.launch`)
WORKS on the stable `torch==2.9.0`/`torch_xla==2.9.0` pin** — confirmed directly on a fresh `v4-8`
node (2026-08-23, `tiny` preset, `--no_torch_compile`, `--no_flash_attention`): `ps aux` during the
run showed 4 real `multiprocessing.spawn_main` worker processes, each with steadily climbing CPU
time (not the stuck-at-launch symptom below), a 2000-step run completed cleanly (`TRAIN_EXIT=0`,
val_bpb converged 7.7→2.46), and the log correctly reported `world_size=4 global_batch=64` (after
fixing a real bug found along the way: `world_size` was computed via
`xr.addressable_runtime_device_count()`, which returns the *calling process's own* local device
count — 1, inside an already-spawned worker — not the total across all processes; fixed to
`xr.world_size()`, the actual global replica count). **Previously reported as "confirmed broken"
below — that finding was specific to the nightly `torch_xla==2.10.0.dev0` build, not a bug in this
project's multichip wiring**, exactly as the static-review hypothesis from 2026-08-22 predicted.
**`--use_flash_attention` + `--multichip` together: confirmed hanging (2026-08-23, tpu5, fresh
v4-8 node, nightly build).** A `--steps 5` smoke test showed all 4 worker processes' cumulative
CPU *time* (via `ps -o time`, not the noisier `%cpu` column) flat across repeated snapshots ~20s
apart (`00:00:18`/`00:00:19`, unchanged) — the exact "spin up then go idle, no further progress"
signature described below for the plain nightly-build hang. Standalone `flash_attention()` calls
work fine on this same node/build (verified via the smoke-test snippet above, max abs diff
~0.011, normal variance) — the hang is specific to combining the two mechanisms, not a broken
install. Conclusion: `--multichip` and `--use_flash_attention` are not usable together on any
build tried so far (stable pin can't do flash-attention at all; nightly can do flash-attention
alone or multichip alone, but not both at once). Use `TPU_VISIBLE_CHIPS`-based independent
single-chip processes (see below) if both flash-attention and multi-chip utilization are wanted
simultaneously.

**Nightly-build-specific hang** (`torch_xla==2.10.0.dev0`, needed only for
`--use_flash_attention`, see above) — kept for reference in case the nightly build is used again:
all worker processes spin up then go idle (CPU time stops climbing) with no further progress,
consistent with a stuck PJRT multi-process rendezvous specific to that build. Static code review
(2026-08-22) found nothing wrong in this project's own collective usage (every rank reaches
`optimizer_step`'s all-reduce in lockstep every step, no other collective appears anywhere in the
file) — consistent with the now-confirmed conclusion that the bug lives in the nightly PJRT
client bootstrap itself, not in `qcute.bytelm_tpu`.

**What else works**: independent single-chip processes via `TPU_VISIBLE_CHIPS`, confirmed directly
— two concurrent single-device processes with `TPU_VISIBLE_CHIPS=0` and `TPU_VISIBLE_CHIPS=1` both
ran and computed correctly with no conflict, no collectives, no rendezvous needed. This is
embarrassingly-parallel, not larger-batch data-parallel: each process is a fully independent
training run (own `--run_name`, own logs/checkpoints), and it's on you to make each process's
`--run_name` distinct (a collision silently interleaves two runs' log lines into one file). Check
the real device count first — a "v4-8" slice's "8" is TensorCores, not addressable devices (v4
runs 2 cores per chip fused as one "megacore" logical device by default): confirmed 4 addressable
devices on this session's `v4-8` node via three independent sources —
`torch_xla.runtime.addressable_runtime_device_count()`, `ls /dev/accel*`, and `tpu-info`. Launch
one process per chip:

**One named `tmux` session per chip** (not 4 background `&` jobs in a single shell — those all die
together if that one shell's session ends; separate sessions survive and can be
attached/detached/killed independently), each with a distinct `--run_name` and its own
`--config` (a 4-way hparam sweep, say — this is what a genuinely useful use of 4 chips on one host
looks like, since `--multichip` doesn't work):

```bash
ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine \
  muaz@<external_ip> "
  tmux new-session -d -s sweep_a 'cd ~/qcute && source .venv/bin/activate && \
    export LD_LIBRARY_PATH=/home/muaz/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/lib && \
    export TPU_VISIBLE_CHIPS=0 && \
    python3 -m qcute.bytelm_tpu --config configs/bytelm_tpu/<variant_a>.py --device xla; echo TRAIN_EXIT=\$?; exec bash'
  tmux new-session -d -s sweep_b 'cd ~/qcute && source .venv/bin/activate && \
    export LD_LIBRARY_PATH=/home/muaz/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/lib && \
    export TPU_VISIBLE_CHIPS=1 && \
    python3 -m qcute.bytelm_tpu --config configs/bytelm_tpu/<variant_b>.py --device xla; echo TRAIN_EXIT=\$?; exec bash'
  # ... one more tmux new-session block per remaining chip (TPU_VISIBLE_CHIPS=2, =3, ...)
"
```

List and attach to any of them from the local machine:

```bash
ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine \
  muaz@<external_ip> "tmux ls"                    # list all sessions on the node

ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine \
  -t muaz@<external_ip> "tmux attach -t sweep_a"   # attach to one (Ctrl-b d to detach)
```

or peek at each without attaching, same `tmux capture-pane -t <session> -p -S -N` pattern as the
single-run case above, once per session name.

## Rough sizing estimate: model/context for 1.0 bpb in 12h on a v4-16 pod

Back-of-envelope only (2026-08-22) — not fit from this project's own scaling data, no controlled
sweep has been run yet. Treat as a starting guess, re-derive once real throughput numbers exist
at a larger model scale.

- v4-16 = 16 TensorCores = **8 addressable chips** (megacore fuses 2 cores/chip, confirmed on the
  `v4-8` node this session: 4 addressable devices via `addressable_runtime_device_count()`/
  `/dev/accel*`/`tpu-info`).
- Compute budget: `8 chips * 275 TFLOPS/chip (bf16 peak) * 35% assumed MFU * 12h = ~3.3e19 FLOPs`.
- `6*N*D` heuristic (N=params, D=bytes trained on) → `N*D <= ~5.5e18`.
- Anchoring to published near-1.0-bpc enwik8 results (Transformer-XL-large 277M, Perceiver AR
  358M/context=8192, ~0.97-0.99 bpc): `N~250-350M`, `D~1-2e10` bytes (~150-220 epochs of enwik8's
  ~90M-byte train split) — product ~2.5-3.5e18, fits the budget with headroom.
- **Pareto point: N≈250-300M params, context≈4096-8192, ~180-200 epochs.**

**Critical caveat**: this budget assumes all 8 chips combine into one data-parallel run via
`--multichip`. Update 2026-08-23: `--multichip` is now **confirmed working** on the stable
`torch==2.9.0`/`torch_xla==2.9.0` pin (see "Optional: multiple TPU chips on one host" below) — the
8-chip budget above is achievable in principle, not blocked on a broken collective anymore. On one
chip alone the same math gives `~4.1e18 FLOPs` → `N*D<=6.9e17`, roughly 4-5x short of the anchor
above; realistic single-chip outcome in 12h is closer to `N≈60-90M`, landing bpb ~1.1-1.2, not
confidently 1.0. Still untested: `--multichip` + `--use_flash_attention` together (nightly-only,
needed for the flash-attention memory savings this doc's zero_kv_sink section below relies on).

## zero_kv_sink + flash-attention: investigation (2026-08-23)

`qcute.bytelm_tpu`'s `zero_kv_sink` (default on) prepends one all-zero, always-attendable K/V
token before every real token, freshly re-concatenated at **every attention layer, every step**
(a static, non-learned attention sink, not a learnable BOS token). Combining it with
`--use_flash_attention` turned out to be a multi-stage investigation, ending in: **don't** —
plain flash-attention without the sink wins on every axis tried.

**Stage 1 — naive combination crashes.** The Pallas kernel's `causal=True` assumes `q_len==kv_len`
(diagonal-aligned); the sink makes `kv_len=T+1` while `q_len=T`, silently misaligning the mask
(confirmed: max abs diff ~2.6 vs. the explicit-mask SDPA reference — not the ~0.01 normal
flash-attention variance, a real correctness bug, not noise). `CausalSelfAttention.forward` (in
`qcute/bytelm_tpu.py`) used to just always fall back to the O(T²)-memory explicit-mask SDPA path
whenever `zero_kv_sink` was on, regardless of the flash flag.

**Stage 2 — the "square" fix.** Padding `q` with one dummy leading row (`q_padded =
cat([zero, q])`) makes `q_len==kv_len==T+1` exactly, restoring correctness — verified against the
explicit-mask reference (max abs diff ~0.007-0.01, matching flash-attention's normal algorithmic
variance). But the kernel *also* requires that common length divisible by its internal block size
(`1024`, observed via the exact error message `q_seq_len=N should be divisible by block_q_dq=1024`)
— so this only works when `context+1` is itself a multiple of 1024 (e.g. `context=8191`, not
`8192`). Implemented in `CausalSelfAttention.forward`, gated on `(T+1) % 1024 == 0`; falls back to
the explicit-mask path otherwise (a real bug was found and fixed here too: the fallback check
originally lived only in `main()`'s startup warning, which printed the warning but never actually
disabled the flash path, so a misconfigured `context` crashed instead of gracefully falling back
— fixed by moving the check into `forward()` itself).

**Stage 3 — it's correct but ~25x too slow.** At `context=8191, batch=4`: **10.0-10.06s/it**,
vs. **0.4s/it** for plain flash-attention without the sink — confirmed identical slowdown with and
without `torch.compile` (10.01s/it uncompiled, 12.2-24s/it compiled, `torch.compile` actually made
it *worse*, not better), and confirmed a pre-allocated `register_buffer` for the zero tensor
(avoiding a fresh `torch.zeros(...)` allocation every call) made **no measurable difference**
(10.06s/it either way) — so the cost is the `torch.cat` itself materializing new
`[B,H,T+1,hd]` K/V/Q tensors at every layer, every step, not allocation overhead or compile
strategy.

**Stage 4 — cheaper alternatives considered, none viable without changing what the sink is.**
- Moving the padding to the *end* of Q instead of the front (to avoid slicing) is **not just
  slower, it's incorrect**: with the sink at kv-index 0 and real queries kept at their original
  (unshifted) indices, every real query systematically misses attending to its own current-byte
  key (worked example: for `T=3`, query 0 attends only `{sink}`, query 1 only `{sink, k0}`, etc.
  — never including its own key). The front-padding version is the only one that's both
  length-compatible and correct.
- A true zero-cost "attention sink" (the off-by-one-softmax / additive-denominator-bias trick from
  the StreamingLLM-style literature — no extra K/V token at all, just a constant added to the
  softmax denominator) isn't implementable through this kernel's exposed API (`ab`/`segment_ids`
  operate on the existing q/kv grid, not on the softmax normalizer).
- The genuinely cheap restructuring — prepend the sink **once**, at the input embedding stage
  (like a normal learnable BOS/CLS token), not re-injected per-layer — uses the plain,
  already-fast flash-attention path unchanged. But it changes what the mechanism *is*: a
  per-layer static zero anchor becomes a single learnable token that evolves through the residual
  stream like any other position. Not implemented (out of scope of the investigation, would need
  `ByteLM.forward`-level restructuring, not just the attention module) — worth doing later if the
  sink's benefit is ever actually verified to matter for convergence.

**Stage 5 — a from-scratch JAX reimplementation didn't rescue it either.** `qcute/bytelm_jax.py`
(pure JAX, same architecture, same enwik8 data, single `jax.jit`-compiled train step — the
natural JAX pattern, unlike torch_xla's per-step lazy-graph tracing plus a Pallas-kernel
`jax.jit` re-invocation from *inside* a torch custom op on every call, visible in the crash
traceback from Stage 1) measured **~3.03s/it** for the sink+flash combination — ~3.3x faster than
torch_xla's 10s/it, suggesting the torch_xla wrapper itself carries real overhead. But a properly
controlled follow-up (sink on/off × fp32/bf16, all 4 combinations) showed **the sink made no
measurable difference in JAX at all** — 0.330, 0.332, and 0.350 it/s across all four variants,
all within noise of each other. So the initial "3.3x faster, confirms the wrapper is the problem"
read was wrong/confounded: JAX's own ~3s/it floor has some other, unidentified cause unrelated to
the sink (candidates not yet investigated: host/device pipelining, the flash kernel's behavior
under `jax.jit` tracing for this exact shape — genuinely unresolved, not chased further since it's
tangential to the main goal).

**Bottom line**: no variant of `zero_kv_sink`-with-flash-attention (torch_xla or JAX, any dtype)
beat plain flash-attention without the sink (0.4s/it on torch_xla) at this model scale. Use
`--no_zero_kv_sink` when `--use_flash_attention` is on, or accept `context=1024*k-1` and the
~25x throughput cost if the sink's architectural benefit is later confirmed to matter enough to
justify it.

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

**If that command suddenly can't connect** (hangs, or fails immediately with `Connection
refused`/`No route to host`) **on a node that was working at the last check-in, check for
preemption before anything else** — every TPU here is a spot instance (see [TPU.md](../TPU.md)),
reclaimable with no warning mid-run, and a dead node is indistinguishable from a flaky connection
until you check state:
`gcloud compute tpus queued-resources describe <qr-name> --project raden-tpu --zone <zone> --format="value(state.state)"`.
`PREEMPTED` means the node and everything on it (the training process, anything not already
copied back) is gone for good — report it and ask how to proceed rather than retrying the
connection or standing up a replacement node unasked. Full detail:
[docs/tpu_direct_ssh.md](tpu_direct_ssh.md)'s caveats section.

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
| `RuntimeError: Unexpected PJRT_ExecuteOptions size: expected 112, got 80` | libtpu bumped independently of the torch/torch_xla pin — plugin/framework PJRT API versions disagree | use the matched nightly build (see above), don't mix a newer libtpu with the stable pin |
| `RuntimeError: Pallas TPU requires a recent libtpu version` | `--use_flash_attention` on the stable pin (`libtpu==0.0.21` < required `0.0.44`) | use the nightly build (see above) |
| `ModuleNotFoundError: No module named 'jax'` calling `flash_attention` | jax not installed — it's the flash-attention kernel's implementation dependency, not a normal project dep | `uv pip install jax` (in the nightly venv) |
| `ImportError: libopenblas.so.0` on the nightly torch wheel | missing system package (the stable release doesn't need it) | `sudo apt-get install -y libopenblas0` |
| `error: ... Wheel version does not match filename` installing a nightly wheel via uv | known nightly-build wheel-metadata quirk, not real corruption | `UV_SKIP_WHEEL_FILENAME_CHECK=1` |
| `--multichip` hangs (workers spin up, CPU time stops climbing, no progress) | stuck PJRT multi-process rendezvous, unresolved on this session's nightly+v4-8 combo | don't use `--multichip`; use independent single-chip processes via `TPU_VISIBLE_CHIPS` instead (see above) |
| `RuntimeError: TPU initialization failed: ... Device or resource busy` | a previous process (stuck, killed slow, or just concurrent) still holds the chip | `ps aux \| grep python3`, `kill -9` the stale PID, retry — `TPU_VISIBLE_CHIPS` conflicts show the same error if two processes claim the same chip index |
