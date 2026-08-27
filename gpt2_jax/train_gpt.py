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
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

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
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-steps", type=int, default=None, help="override computed max_steps (for smoke tests)")
    p.add_argument("--lr", type=float, default=None, help="override MAX_LR (peak learning rate); MIN_LR stays 0.1x this")
    p.add_argument("--run-name", type=str, default=None, help="override the auto-derived run_name (else derived from --config/model/pos-method)")
    p.add_argument("--use-flash-attention", action="store_true",
                    help="use JAX's Pallas TPU flash-attention kernel in place of plain materialized attention "
                         "(this port's own addition, not in Cable's original -- see model_gpt.py's ModelConfig docstring)")

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

    def grad_step(params, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        loss = jax.lax.pmean(loss, "batch")
        grads = jax.lax.pmean(grads, "batch")
        return loss, grads

    p_grad_step = jax.pmap(grad_step, axis_name="batch")

    def apply_step(params, opt_state, grads):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    p_apply_step = jax.pmap(apply_step, axis_name="batch")

    def eval_step(params, batch):
        loss = loss_fn(params, batch)
        return jax.lax.pmean(loss, "batch")

    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    def fetch_pmap_batch(loader: DataLoaderLite):
        # Each of the n_devices gets its own independently-advanced slice of the SAME loader --
        # sequential-not-random, so this just pulls n_devices consecutive B*T windows.
        xs, ys = [], []
        for _ in range(n_devices):
            x, y = loader.next_batch()
            xs.append(np.asarray(x))
            ys.append(np.asarray(y))
        return jnp.stack(xs), jnp.stack(ys)

    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        if step % args.eval_every == 0 or last_step:
            val_loader.reset()
            val_loss = 0.0
            for _ in range(args.eval_steps):
                batch = fetch_pmap_batch(val_loader)
                val_loss += float(p_eval_step(params, batch)[0]) / args.eval_steps
            log(step=step, split="val", loss=val_loss, ppl=math.exp(min(val_loss, 20)))

        loss_accum = 0.0
        grads_accum = None
        for _ in range(grad_accum_steps):
            batch = fetch_pmap_batch(train_loader)
            loss, grads = p_grad_step(params, batch)
            loss_accum += float(loss[0]) / grad_accum_steps
            scaled = jax.tree.map(lambda g: g / grad_accum_steps, grads)
            grads_accum = scaled if grads_accum is None else jax.tree.map(jnp.add, grads_accum, scaled)
        params, opt_state = p_apply_step(params, opt_state, grads_accum)

        dt = time.time() - t0
        tok_per_sec = (B * T * grad_accum_steps * n_devices) / dt
        lr = float(schedule(step))
        log(step=step, split="train", loss=loss_accum, lr=lr, dt_ms=dt * 1000, tok_per_sec=tok_per_sec)

        if last_step:
            single = jax.tree.map(lambda x: x[0], params)
            ckpt_path = (log_dir / f"model_{step}").resolve()
            with ocp.PyTreeCheckpointer() as ckptr:
                ckptr.save(ckpt_path, nnx.to_pure_dict(single))
            log(step=step, event="checkpoint", path=str(ckpt_path))


if __name__ == "__main__":
    main()
