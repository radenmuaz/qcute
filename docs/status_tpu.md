# TPU run status

Living doc — check here rather than assuming anything elsewhere is current. Update in place (don't append) on every run start/stop/change. See [tpu_setup.md](tpu_setup.md) / [tpu_direct_ssh.md](tpu_direct_ssh.md) for access, [TPU.md](../TPU.md) for the queued-resource list (don't create new ones).

## Active runs (2026-09-21)

Zone `us-central2-b`, project `raden-tpu`. tpu3–8 are slated for deletion.

| node | IP | run | tmux | config |
|---|---|---|---|---|
| tpu34a (worker 0) | 35.186.86.22 | `imagenet64_6` | `imagenet64_6` | `image_lagcodec/configs/imagenet64_6.py` |
| tpu34b (worker 1) | 35.186.115.139 | `imagenet64_6` | `imagenet64_6` | same (multihost, run on both hosts) |
| tpu2 | 35.186.15.67 | idle | — | — |

- tpu34 = v4-16 (8 chips, 2 hosts). Both hosts run the same command; log `~/imagenet64_6.log` on each. Launched 2026-09-21 11:32, ETA ~18:20.
- ImageNet64 shards live in `/dev/shm/imagenet64` on each host (copied node-to-node over internal IP, ~16 GB each; gone on reboot/preemption).
- On the nodes use `.venv/bin/python -m image_lagcodec.run_lagcodec ...` (`uv` is not on PATH); `tpu-info` at `~/qcute/.venv/bin/tpu-info`.

## Access

tpu34: `ssh -o ControlPath=~/.ssh/controlmasters/tpu34a-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@35.186.86.22` (and `tpu34b` / `35.186.115.139`).


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
- `tpu-info` HBM (15–27 GiB/chip on these runs) is real usage, not preallocation. Batch/remat changes fail at first compile with `RESOURCE_EXHAUSTED`: d=512 `remat_level` batch 8/12/16 OOM in the older config, per-block `remat` batch 4 fits.
- To stop: `tmux send-keys -t <run> C-c`, then kill only the `.venv/bin/python -m image_lagcodec` PIDs; `pkill -f` on the command matches the ssh line itself.
