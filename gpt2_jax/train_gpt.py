"""JAX port of Cable/src/train_gpt.py's training loop -- same hyperparameters/schedule/optimizer
(AdamW, weight_decay=0.1 on ndim>=2 params only, betas=(0.9,0.95), eps=1e-8, grad-norm-clip 1.0,
linear warmup + cosine decay to 0.1*peak_lr, total_batch_size in tokens with grad-accumulation),
adapted for multi-device TPU data parallelism via jax.pmap across all local devices (in place of
Cable's single/multi-GPU DDP) -- pass --batch-size as the PER-DEVICE micro batch, effective
batch = batch_size * sequence_length * grad_accum_steps * jax.local_device_count().

    uv run python gpt2_jax/train_gpt.py --model tiny --pos-method rope --dataset-dir datasets/fineweb-edu-10B
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm

from data_loader import DataLoaderLite
from model_gpt import Model, ModelConfig, cross_entropy_loss, _HAS_FLASH_ATTENTION

MODEL_SHAPES = {
    # (n_layer, n_head, n_embd) -- identical to Cable's train_gpt.py per-size table
    "large": dict(n_layer=36, n_head=20, n_embd=1280),
    "medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "small": dict(n_layer=12, n_head=12, n_embd=768),
    "tiny": dict(n_layer=6, n_head=8, n_embd=512),
}
TOTAL_BATCH_SIZE_DEFAULT = 2**19  # 524,288 tokens, matches Cable's fineweb-edu-10B default
MICRO_BATCH_SIZES = {"tiny": 64, "small": 32, "medium": 16, "large": 8}
NUM_EPOCHS_DEFAULT = {"tiny": 1, "small": 1, "medium": 1, "large": 1}
NUM_DATASET_TOKENS = 10_000_000_000  # fineweb-edu-10B
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 715


def build_optimizer(max_steps: int):
    schedule = optax.warmup_cosine_decay_schedule(
        # decay_steps is optax's TOTAL schedule length including warmup (not the post-warmup
        # remainder) -- matches Cable's get_lr, which cosine-decays over (max_steps - warmup)
        # steps starting right after warmup ends, i.e. reaches end_value exactly at max_steps.
        init_value=0.0, peak_value=MAX_LR, warmup_steps=WARMUP_STEPS,
        decay_steps=max(WARMUP_STEPS + 1, max_steps), end_value=MIN_LR,
    )
    # Known one-step warmup offset vs. Cable: its get_lr uses max_lr*(step+1)/warmup_steps
    # (nonzero at step 0, peak at step 714); this schedule is 0 at step 0, peak at step 715 -- negligible over a full run, not fixed.
    # weight_decay applies to ndim>=2 params only (matmul/embedding weights), not biases/norms --
    # matches Cable's configure_optimizers' decay_params/nodecay_params split exactly.
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
    pre.add_argument("--config", type=Path, default=None, help="Python config file (configs/gpt2_jax/*.py); CLI flags override it")
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        description="Cable model_gpt.py, ported to JAX (rope/learnable/base pos_methods only)", parents=[pre]
    )
    p.add_argument("--model", choices=list(MODEL_SHAPES), default="tiny")
    p.add_argument("--pos-method", choices=["rope", "learnable", "base"], default="rope")
    p.add_argument("--dataset-dir", type=str, default=None, help="dir of train_*.npy/val_*.npy shards from dataset_preparation.py")
    p.add_argument("--save-dir", type=str, default="gpt2_jax/logs")
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--total-batch-size", type=int, default=TOTAL_BATCH_SIZE_DEFAULT)
    p.add_argument("--batch-size", type=int, default=None, help="per-device micro batch size")
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-steps", type=int, default=None, help="override computed max_steps (for smoke tests)")
    p.add_argument("--lr", type=float, default=None, help="override MAX_LR (peak learning rate); MIN_LR stays 0.1x this")
    p.add_argument("--run-name", type=str, default=None, help="override the auto-derived run_name (else derived from --config/model/pos-method)")
    p.add_argument("--use-flash-attention", action="store_true")

    if pre_args.config:
        config_vars = load_config_module(pre_args.config)
        defaults_before = {a.dest: a.default for a in p._actions}
        known = set(defaults_before)
        unknown = sorted(set(config_vars) - known)
        recognized = {k: v for k, v in config_vars.items() if k in known}
        added = sorted(k for k, v in recognized.items() if defaults_before[k] is None and v is not None)
        updated = sorted(k for k, v in recognized.items() if k not in added and v != defaults_before[k])
        print(f"config: {pre_args.config}  parsed={recognized}")
        if added:
            print(f"config: added (was unset) -> {', '.join(added)}")
        if updated:
            print(f"config: updated (overrides a hardcoded default) -> {', '.join(updated)}")
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
    B = args.batch_size or MICRO_BATCH_SIZES[args.model]
    T = args.sequence_length
    num_epochs = args.num_epochs or NUM_EPOCHS_DEFAULT[args.model]
    total_batch_size = args.total_batch_size
    assert total_batch_size % (B * T * n_devices) == 0, "total_batch_size must be divisible by batch_size * sequence_length * n_devices"
    grad_accum_steps = total_batch_size // (B * T * n_devices)
    max_steps = args.max_steps or num_epochs * (NUM_DATASET_TOKENS // total_batch_size)

    if args.run_name:
        run_name = args.run_name
    elif pre_args.config:
        run_name = pre_args.config.stem
    else:
        run_name = f"{args.model}_{args.pos_method}_fineweb-edu-10B_{num_epochs}_{total_batch_size}_{B}_{T}"
    log_dir = Path(args.save_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot the fully-resolved config (config file + CLI overrides + hardcoded defaults) into
    # the run's own log dir -- reproducible/inspectable independent of what --config pointed at,
    # and itself loadable via --config for an exact rerun.
    resolved_lines = [f"{k} = {v!r}" for k, v in sorted(vars(args).items()) if k != "config"]
    (log_dir / "resolved_config.py").write_text("\n".join(resolved_lines) + "\n")
    log_f = open(log_dir / "log.jsonl", "a")

    def log(**record):
        record["t"] = time.time()
        print(record)
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    print(f"n_devices={n_devices}  batch_size(per-device)={B}  seq_len={T}  grad_accum_steps={grad_accum_steps}  "
          f"total_batch_size={total_batch_size}  max_steps={max_steps}  pos_method={args.pos_method}  "
          f"use_flash_attention={args.use_flash_attention} (available={_HAS_FLASH_ATTENTION})")

    train_loader = DataLoaderLite(B, T, 0, 1, "train", args.dataset_dir)
    val_loader = DataLoaderLite(B, T, 0, 1, "val", args.dataset_dir)

    cfg = ModelConfig(pos_method=args.pos_method, block_size=T, vocab_size=50304,
                       use_flash_attention=args.use_flash_attention, **MODEL_SHAPES[args.model])
    rngs = nnx.Rngs(args.seed)
    model = Model(cfg, rngs=rngs)
    graphdef, params = nnx.split(model)
    num_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"num_params={num_params:,}")

    optimizer, schedule = build_optimizer(max_steps)
    opt_state = optimizer.init(params)

    # jax.device_put_replicated was removed -- replicate manually: stack n_devices copies along
    # a new leading axis, then shard that axis one-per-device via a Mesh/NamedSharding. Produces
    # the same per-device-leading-axis layout pmap expects.
    devices = jax.local_devices()
    mesh = jax.sharding.Mesh(devices, ("d",))
    replicate_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))

    def replicate(pytree):
        stacked = jax.tree.map(lambda x: jnp.stack([x] * len(devices)), pytree)
        return jax.tree.map(lambda x: jax.device_put(x, replicate_sharding), stacked)

    params = replicate(params)
    opt_state = replicate(opt_state)

    def loss_fn(params, batch):
        model = nnx.merge(graphdef, params)
        x, y = batch
        logits = model(x)
        return cross_entropy_loss(logits, y)

    # Fused grad-accum: the whole microbatch loop runs as one jax.lax.scan inside a single
    # pmapped call (instead of a Python loop of separate pmap calls each read back to host via
    # float()) -- avoids grad_accum_steps blocking host<->device syncs per training step, letting
    # XLA see/optimize the whole step as one graph and keeping dispatch async.
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
        zero_loss = jnp.zeros((), dtype=jnp.float32)
        (loss_sum, grads_sum), _ = jax.lax.scan(scan_fn, (zero_loss, zero_grads), (x_all, y_all))
        loss_mean = loss_sum / grad_accum_steps
        grads_mean = jax.tree.map(lambda g: g / grad_accum_steps, grads_sum)
        loss_mean = jax.lax.pmean(loss_mean, "batch")
        grads_mean = jax.lax.pmean(grads_mean, "batch")
        return loss_mean, grads_mean

    p_grad_accum_step = jax.pmap(grad_accum_step, axis_name="batch")

    def apply_step(params, opt_state, grads):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # donate params/opt_state buffers: after this call the old copies are dead (superseded by
    # the returned new params/opt_state), so XLA can reuse their device memory in place instead
    # of allocating fresh buffers every step.
    p_apply_step = jax.pmap(apply_step, axis_name="batch", donate_argnums=(0, 1))

    def eval_accum_step(params, batch):
        x_all, y_all = batch  # per-device: [eval_steps, B, T]

        def scan_fn(loss_sum, xy):
            x, y = xy
            return loss_sum + loss_fn(params, (x, y)), None

        loss_sum, _ = jax.lax.scan(scan_fn, jnp.zeros((), dtype=jnp.float32), (x_all, y_all))
        loss_mean = loss_sum / x_all.shape[0]
        return jax.lax.pmean(loss_mean, "batch")

    p_eval_accum_step = jax.pmap(eval_accum_step, axis_name="batch")

    def fetch_pmap_batch(loader: DataLoaderLite, steps: int):
        # Builds one combined [n_devices, steps, B, T] batch (steps = grad_accum_steps or
        # eval_steps) so the whole accumulation loop above can run as a single scan/pmap call --
        # each of the n_devices gets its own independently-advanced slice of the SAME loader
        # (sequential-not-random), consistent across the `steps` dimension.
        xs = np.empty((n_devices, steps, B, T), dtype=np.int32)
        ys = np.empty((n_devices, steps, B, T), dtype=np.int32)
        for s in range(steps):
            for d in range(n_devices):
                x, y = loader.next_batch()
                xs[d, s] = np.asarray(x)
                ys[d, s] = np.asarray(y)
        return jnp.asarray(xs), jnp.asarray(ys)

    # Background prefetch: build next step's training batch (host-side numpy/memmap work) on a
    # separate thread while the device is still executing the current step's pmap call, instead
    # of blocking the main loop on fetch_pmap_batch between every step.
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


if __name__ == "__main__":
    main()
