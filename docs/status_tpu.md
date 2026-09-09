# TPU run status

Living doc — check here rather than assuming anything elsewhere is current. Update in place (don't append) on every run start/stop/change. See [tpu_setup.md](tpu_setup.md) / [tpu_direct_ssh.md](tpu_direct_ssh.md) for access, [TPU.md](../TPU.md) for the queued-resource list (don't create new ones).

## Active runs (2026-09-09)

All 4 nodes are v4-8 (4 chips/device each), zone `us-central2-b`, project `raden-tpu`.

| node | external IP | run | tmux session | config |
|---|---|---|---|---|
| tpu1 | 35.186.98.243 | `cifar10_stack_lag0` | `cifar10_stack_lag0` | `image_lagcodec/configs/cifar10_stack_lag0.py` (lag=0) |
| tpu2 | 35.186.15.67 | `cifar10_stack_lag4` | `cifar10_stack_lag4` | `image_lagcodec/configs/cifar10_stack_lag4.py` (lag=4) |
| tpu3 | 107.167.160.20 | `cifar10_stack_lag16` | `cifar10_stack_lag16` | `image_lagcodec/configs/cifar10_stack_lag16.py` (lag=16) |
| tpu4 | 35.186.33.7 | `cifar10_stack_lagmax` | `cifar10_stack_lagmax` | `image_lagcodec/configs/cifar10_stack_lagmax.py` (lag=255/max) |

- All 4 launched via `tmux new-session -d -s <run_name> '... | tee ~/<run_name>.log; exec bash'` (log tee'd for periodic `scp` pulls, per convention).
- Status as of launch: epoch 1-2/100, ~1.6s/it, 195 steps/epoch (batch_size=64/device x4 devices = 256 effective), no crashes. First eval/qual-gen/checkpoint at epoch 10 (`eval_every_epochs=10`).
- Full architecture/config context: [status_image_lagcodec.md](status_image_lagcodec.md).

## Access

```
ssh -o ControlPath=~/.ssh/controlmasters/tpu1-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.98.243
ssh -o ControlPath=~/.ssh/controlmasters/tpu2-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.15.67
ssh -o ControlPath=~/.ssh/controlmasters/tpu3-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@107.167.160.20
ssh -o ControlPath=~/.ssh/controlmasters/tpu4-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.33.7
```
`tmux attach -t <run_name>` / `tmux capture-pane -t <run_name> -p -S -N` per node once connected. To stop a run: `tmux send-keys -t <run_name> C-c` (Ctrl-C first, per convention — `tmux kill-session` alone can leave the python process holding the TPU lock).

## Known gotchas hit this session (see status_image_lagcodec.md for full detail)

- `tmux kill-session` does not reliably kill the python process holding libtpu — verify with `pgrep -af 'python3 -m image_lagcodec'` before relaunching, or the new process fails with `RuntimeError: Unable to initialize backend 'tpu': ABORTED: The TPU is already in use`.
- `tpu-info`'s HBM usage reads ~100% on every chip during training — this is JAX's default `XLA_PYTHON_CLIENT_PREALLOCATE=true` grabbing the whole device upfront, not a real memory-pressure signal. Use `jax.local_devices()[i].memory_stats()["bytes_in_use"]` for actual live usage.
- Batch size is per-device in these configs; effective global batch is `batch_size * n_devices` (4 here).
