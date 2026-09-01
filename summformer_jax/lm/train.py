"""Training loop for the summformer.py-based lm lineage -- mirrors gpt2_jax/train_gpt.py's
optimizer/schedule/dataloader/batch semantics EXACTLY (same constants, same DataLoaderLite access
pattern via lm/data_loader.py, same defaults), so any loss/bpb difference between this and
gpt2_jax's own runs isolates the architecture change, not a training-recipe difference. Config file
exposes a `build_summformer(rngs) -> ARHead` factory (see configs/thin512_win16_allfuse8_chained.py)
instead of gpt2_jax's ModelConfig/Model, since the new architecture composes Embedder/Encoder/
Decoder objects rather than a single dataclass -- everything else here is a direct structural port,
not a reinterpretation (written from a clean slate 2026-08-31 after the previous version diverged
from the baseline in several unauthorized ways: random-sampling dataloader instead of DataLoaderLite's
sequential access, eval_every=954 instead of the real default 250, seed=0 instead of 1234).

    uv run python summformer_jax/lm/train.py --config configs/thin512_win16_allfuse8_chained.py --dataset-dir data/fineweb-edu-10B
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # summformer_jax/

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm

from data_loader import DataLoaderLite

TOTAL_BATCH_SIZE_DEFAULT = 2**19  # 524,288 tokens, matches gpt2_jax/Cable's fineweb-edu-10B default
NUM_DATASET_TOKENS = 10_000_000_000  # fineweb-edu-10B
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 715


def build_optimizer(max_steps: int):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=MAX_LR, warmup_steps=WARMUP_STEPS,
        decay_steps=max(WARMUP_STEPS + 1, max_steps), end_value=MIN_LR,
    )

    def decay_mask(params):
        return jax.tree.map(lambda p: p.ndim >= 2, params)

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1, mask=decay_mask),
    ), schedule


def load_config_module(path: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None, help="Python config file; CLI flags override it")
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--dataset-dir", type=str, default=None, help="dir of train_*.npy/val_*.npy shards")
    p.add_argument("--save-dir", type=str, default="summformer_jax/lm/logs")
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--total-batch-size", type=int, default=TOTAL_BATCH_SIZE_DEFAULT)
    p.add_argument("--batch-size", type=int, default=None, help="per-device micro batch size")
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-steps", type=int, default=None, help="override computed max_steps (for smoke tests)")
    p.add_argument("--lr", type=float, default=None, help="override MAX_LR; MIN_LR stays 0.1x this")
    p.add_argument("--run-name", type=str, default=None)

    if pre_args.config:
        config_vars = load_config_module(pre_args.config)
        defaults_before = {a.dest: a.default for a in p._actions}
        known = set(defaults_before)
        recognized = {k: v for k, v in config_vars.items() if k in known}
        unknown = sorted(set(config_vars) - known)
        print(f"config: {pre_args.config}  parsed={recognized}")
        if unknown:
            print(f"config: WARNING unrecognized keys, ignored -> {', '.join(unknown)}")
        p.set_defaults(**recognized)
    args = p.parse_args()
    if args.dataset_dir is None:
        p.error("--dataset-dir is required (directly or via --config)")

    global MAX_LR, MIN_LR
    if args.lr is not None:
        MAX_LR = args.lr
        MIN_LR = args.lr * 0.1

    n_devices = jax.local_device_count()
    B = args.batch_size
    T = args.sequence_length
    total_batch_size = args.total_batch_size
    assert total_batch_size % (B * T * n_devices) == 0, "total_batch_size must be divisible by batch_size * sequence_length * n_devices"
    grad_accum_steps = total_batch_size // (B * T * n_devices)
    max_steps = args.max_steps or args.num_epochs * (NUM_DATASET_TOKENS // total_batch_size)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"lm_{B}_{T}_{total_batch_size}")
    log_dir = Path(args.save_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    resolved_lines = [f"{k} = {v!r}" for k, v in sorted(vars(args).items()) if k != "config"]
    (log_dir / "resolved_config.py").write_text("\n".join(resolved_lines) + "\n")
    log_f = open(log_dir / "log.jsonl", "a")

    def log(**record):
        record["t"] = time.time()
        print(record)
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    print(f"n_devices={n_devices}  batch_size(per-device)={B}  seq_len={T}  grad_accum_steps={grad_accum_steps}  "
          f"total_batch_size={total_batch_size}  max_steps={max_steps}")

    train_loader = DataLoaderLite(B, T, 0, 1, "train", args.dataset_dir)
    val_loader = DataLoaderLite(B, T, 0, 1, "val", args.dataset_dir)

    sys.path.insert(0, str(pre_args.config.parent))
    cfg_module = importlib.import_module(pre_args.config.stem)
    head = cfg_module.build_summformer(nnx.Rngs(args.seed))
    graphdef, params = nnx.split(head)
    num_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"num_params={num_params:,}")

    optimizer, schedule = build_optimizer(max_steps)
    opt_state = optimizer.init(params)

    devices = jax.local_devices()
    mesh = jax.sharding.Mesh(devices, ("d",))
    replicate_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))

    def replicate(pytree):
        stacked = jax.tree.map(lambda x: jnp.stack([x] * len(devices)), pytree)
        return jax.tree.map(lambda x: jax.device_put(x, replicate_sharding), stacked)

    params = replicate(params)
    opt_state = replicate(opt_state)

    def loss_fn(params, batch):
        m = nnx.merge(graphdef, params)
        x, y = batch
        seq = jnp.concatenate([x, y[:, -1:]], axis=-1)  # (B, T+1); ARHead shifts internally
        loss, _metrics = m(seq)  # ARHead's "bpb" is loss/ln(2) -- meaningless for BPE tokens
        return loss

    def grad_accum_step(params, batch):
        x_all, y_all = batch  # per-device: [grad_accum_steps, B, T]

        def scan_fn(carry, xy):
            loss_sum, grads_sum = carry
            x, y = xy
            loss, grads = jax.value_and_grad(loss_fn)(params, (x, y))
            loss_sum = loss_sum + loss
            grads_sum = jax.tree.map(jnp.add, grads_sum, grads)
            return (loss_sum, grads_sum), None

        zero_grads = jax.tree.map(jnp.zeros_like, params)
        zero = jnp.zeros((), dtype=jnp.float32)
        (loss_sum, grads_sum), _ = jax.lax.scan(scan_fn, (zero, zero_grads), (x_all, y_all))
        loss_mean = loss_sum / grad_accum_steps
        grads_mean = jax.tree.map(lambda g: g / grad_accum_steps, grads_sum)
        loss_mean, grads_mean = (jax.lax.pmean(v, "d") for v in (loss_mean, grads_mean))
        return loss_mean, grads_mean

    p_grad_accum_step = jax.pmap(grad_accum_step, axis_name="d")

    def apply_step(params, opt_state, grads):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    p_apply_step = jax.pmap(apply_step, axis_name="d", donate_argnums=(0, 1))

    def eval_accum_step(params, batch):
        x_all, y_all = batch  # per-device: [eval_steps, B, T]

        def scan_fn(loss_sum, xy):
            x, y = xy
            return loss_sum + loss_fn(params, (x, y)), None

        loss_sum, _ = jax.lax.scan(scan_fn, jnp.zeros((), dtype=jnp.float32), (x_all, y_all))
        n = x_all.shape[0]
        return jax.lax.pmean(loss_sum / n, "d")

    p_eval_accum_step = jax.pmap(eval_accum_step, axis_name="d")

    def fetch_pmap_batch(loader: DataLoaderLite, steps: int):
        xs = np.empty((n_devices, steps, B, T), dtype=np.int32)
        ys = np.empty((n_devices, steps, B, T), dtype=np.int32)
        for s in range(steps):
            for d in range(n_devices):
                x, y = loader.next_batch()
                xs[d, s] = np.asarray(x)
                ys[d, s] = np.asarray(y)
        return jnp.asarray(xs), jnp.asarray(ys)

    _prefetch_q: queue.Queue = queue.Queue(maxsize=2)

    def _prefetch_worker():
        while True:
            _prefetch_q.put(fetch_pmap_batch(train_loader, grad_accum_steps))

    threading.Thread(target=_prefetch_worker, daemon=True).start()

    pbar = tqdm(range(max_steps), desc="train", dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    val_loss = None
    for step in pbar:
        t0 = time.time()
        last_step = step == max_steps - 1

        if step % args.eval_every == 0 or last_step:
            val_loader.reset()
            val_batch = fetch_pmap_batch(val_loader, args.eval_steps)
            val_loss = float(p_eval_accum_step(params, val_batch)[0])
            log(step=step, split="val", loss=val_loss, ppl=math.exp(min(val_loss, 20)))

        batch = _prefetch_q.get()
        loss, grads = p_grad_accum_step(params, batch)
        params, opt_state = p_apply_step(params, opt_state, grads)
        loss_accum = float(loss[0])

        dt = time.time() - t0
        tok_per_sec = (B * T * grad_accum_steps * n_devices) / dt
        lr = float(schedule(step))
        log(step=step, split="train", loss=loss_accum, ppl=math.exp(min(loss_accum, 20)), lr=lr,
            dt_ms=dt * 1000, tok_per_sec=tok_per_sec)
        postfix = {"loss": f"{loss_accum:.6f}", "tok/s": f"{tok_per_sec:.2f}", "lr": f"{lr:.8f}"}
        if val_loss is not None:
            postfix["val_loss"] = f"{val_loss:.6f}"
        pbar.set_postfix(postfix)
        if step == 0:
            print(f"[compile] step 0 wall time (includes first-call XLA compile): {dt:.2f}s", flush=True)

        if last_step:
            single = jax.tree.map(lambda x: x[0], params)
            ckpt_path = (log_dir / f"model_{step}").resolve()
            with ocp.PyTreeCheckpointer() as ckptr:
                ckptr.save(ckpt_path, nnx.to_pure_dict(single))
            log(step=step, event="checkpoint", path=str(ckpt_path))

    log(event="done", step=max_steps)


if __name__ == "__main__":
    main()
