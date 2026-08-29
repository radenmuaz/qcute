"""Training loop for image_gen/summformer.py's ConfigV2/SummTransformerV2, reading whole
raster-RGB-byte images via imagenet_dataloader.py. Mirrors gpt2_jax/train_gpt.py's multi-device
pattern closely: jax.pmap data-parallel across every local device (default) or a single device
(--devices one), replicated params/opt_state, fused grad-accumulation via one jax.lax.scan inside
the pmapped call (--total-batch-size images/step, grad_accum_steps = total_batch_size /
(batch_size * n_devices)), a background prefetch thread building the next step's batch while the
device executes the current one, and a pmapped eval sweep (falls back to single-device for the
val split's remainder that doesn't fill a full n_devices-sized group).

--shard-dir omitted -> synthetic random-RGB batches (no real dataset needed).

Every eval (one per --steps-per-epoch) computes a REAL full-validation-set bpd (bits per dimension
-- identical to bits-per-byte here, since each RGB byte is one dimension) sweep over the val split,
AND generates+saves sample images. Sample generation stays single-device/sequential (unreplicated
params) -- generate_kv_cache_fully_static generates one image at a time, doesn't parallelize the
same way training/eval-loss do. Train/val split follows the standard downsampled-ImageNet
convention; --shard-dir must hold both `*train*.npy` and `*validation*.npy` shards
(scripts/prep_imagenet64.py's own output layout).

**Every CLI flag below can also be set from the --config file** (config file value used whenever
the flag itself isn't passed on the command line; an explicit CLI flag always wins) -- config
files are the natural place to pin steps/batch_size/lr/save_every for a specific run, matching
every other lineage in this repo.

**Checkpointing**: --save-every saves params + optimizer state + step + dataloader position to
--checkpoint-dir/<run_name>/step_<N> (orbax PyTreeCheckpointer for params/opt_state, pickle for
the rest -- see checkpoint_io.py). --resume-from restores all of that and continues from the saved
step, including exact train-loader position (shard/row/permutation), so resuming reproduces the
same data order.

    uv run python summformer_jax/image_gen/train.py --config summformer_jax/image_gen/configs/image64_test.py --steps 20
    uv run python summformer_jax/image_gen/train.py --config summformer_jax/image_gen/configs/tiny_1.py
    uv run python summformer_jax/image_gen/train.py --config summformer_jax/image_gen/configs/tiny_1.py --devices one   # single-chip debug
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import queue
import threading
import time
from dataclasses import fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from PIL import Image
from tqdm import tqdm

from summformer import ConfigV2, SummTransformerV2
from imagenet_dataloader import ImageByteLoader
from checkpoint_io import save_checkpoint, load_checkpoint


def load_config_module(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


def build_config(config_vars: dict) -> ConfigV2:
    valid = {f.name for f in fields(ConfigV2)}
    return ConfigV2(**{k: v for k, v in config_vars.items() if k in valid})


def replicate(pytree, devices):
    mesh = jax.sharding.Mesh(devices, ("d",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
    stacked = jax.tree.map(lambda x: jnp.stack([x] * len(devices)), pytree)
    return jax.tree.map(lambda x: jax.device_put(x, sharding), stacked)


def unreplicate(pytree):
    return jax.tree.map(lambda x: x[0], pytree)


def full_val_bpd(p_eval_step, eval_step_single, state, single_state, val_loader: ImageByteLoader,
                  batch_size: int, n_devices: int, eval_batches: int | None = None) -> tuple[float, float]:
    """Real sweep over the val split (or the first eval_batches pmap groups, for speed) -- not one
    random batch. Batches are grouped n_devices-at-a-time and evaluated via the pmapped
    `p_eval_step` (on the still-replicated `state`, mirroring gpt2_jax's eval path -- no
    unreplicate needed for a loss-only forward pass); a leftover group smaller than n_devices (the
    tail of an odd-sized val split) falls back to single-device `eval_step_single`. Returns
    (mean_loss, mean_bpd); bpd == bpb here (one RGB byte = one dimension)."""
    losses = []
    group = []
    n_groups = 0
    for batch in tqdm(val_loader.full_sweep(batch_size), desc="val sweep", leave=False):
        group.append(batch)
        if len(group) < n_devices:
            continue
        tokens = jnp.asarray(np.stack(group), dtype=jnp.int32)
        loss = p_eval_step(state, tokens)
        losses.append(float(loss[0]))
        n_groups += 1
        group = []
        if eval_batches is not None and n_groups >= eval_batches:
            group = []
            break
    for batch in group:
        tokens = jnp.asarray(batch, dtype=jnp.int32)
        losses.append(float(eval_step_single(single_state, tokens)))
    mean_loss = sum(losses) / len(losses)
    return mean_loss, mean_loss / math.log(2)


def eval_and_sample(model: SummTransformerV2, p_eval_step, eval_step_single, state, single_state,
                     val_loader: ImageByteLoader, resolution: int, batch_size: int, n_devices: int,
                     n_samples: int, prompt_len: int, eval_batches: int | None,
                     sample_dir: Path, tag: str, seed: int) -> dict:
    """Full-val-set bpd (not one batch), plus n_samples generated images saved to
    sample_dir/<tag>_<i>.png -- each seeded from a REAL val image's first prompt_len bytes
    (completion-style, standard in this literature for qualitative inspection -- a random-byte
    prompt looks like noise, not the start of a coherent image) and drawn via true stochastic
    sampling (temperature=1, not the greedy default -- a qualitative sample should be an actual
    draw from the model's distribution, not the single most-likely, often-repetitive continuation)."""
    val_loss, val_bpd = full_val_bpd(p_eval_step, eval_step_single, state, single_state, val_loader,
                                      batch_size, n_devices, eval_batches)

    sample_dir.mkdir(parents=True, exist_ok=True)
    prompt_batch = val_loader.next_batch()
    key = jax.random.PRNGKey(seed)
    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        prompt = jnp.asarray(prompt_batch[i % prompt_batch.shape[0], :prompt_len], dtype=jnp.int32)
        n_new = model.cfg.context_len - prompt_len
        out = model.generate_kv_cache_fully_static(prompt, n_new, key=subkey, temperature=1.0)
        img = np.array(out[:resolution * resolution * 3]).reshape(resolution, resolution, 3).astype(np.uint8)
        Image.fromarray(img).save(sample_dir / f"{tag}_{i}.png")

    return {"val_loss": val_loss, "val_bpd": val_bpd}


# name, type, default -- every entry here is settable from either the CLI flag (--<name-with-dashes>)
# or the same key in the config file; CLI wins whenever the flag is actually passed.
_OVERRIDABLE = [
    ("resolution", int, 64),
    ("shard_dir", str, None),
    ("steps", int, 20),
    ("steps_per_epoch", int, 10),
    ("eval_samples", int, 1),
    ("eval_prompt_len", int, 16),
    ("eval_batches", int, None),
    ("sample_dir", str, "summformer_jax/image_gen/samples"),
    ("lr", float, 3e-4),
    ("warmup_steps", int, 200),
    ("seed", int, 0),
    ("save_dir", str, "summformer_jax/image_gen/logs"),
    ("batch_size", int, 2),
    ("total_batch_size", int, None, "images/step; defaults to batch_size*n_devices (no grad-accum)"),
    ("devices", str, "all", "'all' local devices (default) or 'one' (single-chip debug)"),
    ("save_every", int, None),
    ("checkpoint_dir", str, "summformer_jax/image_gen/checkpoints"),
    ("resume_from", str, None),
]


def resolve_args(cli_args: argparse.Namespace, config_vars: dict) -> argparse.Namespace:
    """CLI flag (if explicitly passed, i.e. not None) > config file value > hardcoded default."""
    resolved = argparse.Namespace(**vars(cli_args))
    for entry in _OVERRIDABLE:
        name, default = entry[0], entry[2]
        cli_val = getattr(cli_args, name)
        if cli_val is not None:
            continue
        setattr(resolved, name, config_vars.get(name, default))
    return resolved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-name", type=str, default=None, help="defaults to the config file's stem")
    for entry in _OVERRIDABLE:
        name, typ, default = entry[0], entry[1], entry[2]
        extra_help = entry[3] if len(entry) > 3 else ""
        p.add_argument(f"--{name.replace('_', '-')}", type=typ, default=None,
                        help=f"{extra_help} (overridable from config file's `{name}`, default {default!r})")
    cli_args = p.parse_args()

    config_vars = load_config_module(cli_args.config)
    args = resolve_args(cli_args, config_vars)
    cfg = build_config(config_vars)
    batch_size = args.batch_size

    run_name = args.run_name or cli_args.config.stem
    log_dir = Path(args.save_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(log_dir / "log.jsonl", "a")

    def log(*, console=True, **record):
        record["t"] = time.time()
        if console:
            tqdm.write(str(record))
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    devices = jax.local_devices() if args.devices == "all" else jax.local_devices()[:1]
    n_devices = len(devices)

    total_batch_size = args.total_batch_size or (batch_size * n_devices)
    assert total_batch_size % (batch_size * n_devices) == 0, (
        "total_batch_size must be divisible by batch_size * n_devices"
    )
    grad_accum_steps = total_batch_size // (batch_size * n_devices)

    train_loader = ImageByteLoader(batch_size, args.resolution, shard_dir=args.shard_dir, split="train", seed=args.seed)
    val_loader = (
        ImageByteLoader(batch_size, args.resolution, shard_dir=args.shard_dir, split="validation", seed=args.seed + 1)
        if args.shard_dir is not None else train_loader  # synthetic mode has no real split distinction
    )
    assert train_loader.seq_len == cfg.context_len, (
        f"loader seq_len={train_loader.seq_len} (resolution={args.resolution}) != cfg.context_len={cfg.context_len} "
        f"-- config's context_len must equal resolution*resolution*3"
    )

    model = SummTransformerV2(cfg, rngs=nnx.Rngs(args.seed))
    graphdef, state = nnx.split(model)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr, warmup_steps=args.warmup_steps,
        decay_steps=args.steps, end_value=args.lr * 0.1,
    )
    optimizer = optax.adamw(lr_schedule, b1=0.9, b2=0.95, weight_decay=0.1)
    opt_state = optimizer.init(state)

    start_step = 0
    if args.resume_from is not None:
        params_restored, opt_state_restored, extra = load_checkpoint(args.resume_from, nnx.to_pure_dict(state), opt_state)
        nnx.replace_by_pure_dict(state, params_restored)
        opt_state = opt_state_restored
        start_step = extra["step"]
        train_loader.load_state_dict(extra["loader_state"])
        log(event="resumed", step=start_step, path=args.resume_from)

    def loss_fn(state, tokens):
        model = nnx.merge(graphdef, state)
        loss, metrics = model(tokens)
        return loss, metrics

    def eval_step_single_fn(state, tokens):
        loss, _ = loss_fn(state, tokens)
        return loss
    eval_step_single = jax.jit(eval_step_single_fn)

    def eval_step_pmapped_fn(state, tokens):
        loss, _ = loss_fn(state, tokens)
        return jax.lax.pmean(loss, "d")
    p_eval_step = jax.pmap(eval_step_pmapped_fn, axis_name="d")

    # Fused grad-accum: the whole microbatch loop runs as one jax.lax.scan inside a single pmapped
    # call (mirrors gpt2_jax/train_gpt.py's grad_accum_step) -- avoids grad_accum_steps blocking
    # host<->device syncs per training step, letting XLA see/optimize the whole step as one graph.
    def grad_accum_step(state, batch):
        zero_grads = jax.tree.map(jnp.zeros_like, state)

        def scan_fn(carry, tokens):
            loss_sum, bpb_sum, grads_sum = carry
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state, tokens)
            loss_sum = loss_sum + loss
            bpb_sum = bpb_sum + metrics["bpb"]
            grads_sum = jax.tree.map(jnp.add, grads_sum, grads)
            return (loss_sum, bpb_sum, grads_sum), None

        zero = (jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32), zero_grads)
        (loss_sum, bpb_sum, grads_sum), _ = jax.lax.scan(scan_fn, zero, batch)
        loss_mean = loss_sum / grad_accum_steps
        bpb_mean = bpb_sum / grad_accum_steps
        grads_mean = jax.tree.map(lambda g: g / grad_accum_steps, grads_sum)
        loss_mean = jax.lax.pmean(loss_mean, "d")
        bpb_mean = jax.lax.pmean(bpb_mean, "d")
        grads_mean = jax.lax.pmean(grads_mean, "d")
        return loss_mean, bpb_mean, grads_mean

    p_grad_accum_step = jax.pmap(grad_accum_step, axis_name="d")

    def apply_step(state, opt_state, grads):
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state

    p_apply_step = jax.pmap(apply_step, axis_name="d", donate_argnums=(0, 1))

    state = replicate(state, devices)
    opt_state = replicate(opt_state, devices)

    n_train_images = train_loader.n_images
    total_epochs = (args.steps * total_batch_size / n_train_images) if n_train_images else None
    print(f"config: {config_vars}")
    print(f"seq_len={train_loader.seq_len} batch_size(per-device)={batch_size} n_devices={n_devices} "
          f"grad_accum_steps={grad_accum_steps} total_batch_size={total_batch_size} "
          f"synthetic={train_loader.synthetic} run_name={run_name} start_step={start_step}")
    epochs_str = f"{total_epochs:.3f}" if total_epochs is not None else "N/A (synthetic)"
    print(f"n_train_images={n_train_images} steps={args.steps} total_epochs={epochs_str}")

    def fetch_pmap_batch(loader: ImageByteLoader, steps: int):
        """[n_devices, steps, batch_size, seq_len] -- each device gets its own independently-
        advanced slice of the SAME loader, consistent across the `steps` (grad-accum) dimension."""
        xs = np.empty((n_devices, steps, batch_size, loader.seq_len), dtype=np.uint8)
        for s in range(steps):
            for d in range(n_devices):
                xs[d, s] = loader.next_batch()
        return jnp.asarray(xs, dtype=jnp.int32)

    # Background prefetch: build next step's batch (host-side numpy work) on a separate thread
    # while the device is still executing the current step's pmap call, instead of blocking the
    # main loop on fetch_pmap_batch between every step.
    _prefetch_q: queue.Queue = queue.Queue(maxsize=2)

    def _prefetch_worker():
        while True:
            _prefetch_q.put(fetch_pmap_batch(train_loader, grad_accum_steps))

    threading.Thread(target=_prefetch_worker, daemon=True).start()

    def save(step):
        single_state = unreplicate(state)
        single_opt_state = unreplicate(opt_state)
        ckpt_path = Path(args.checkpoint_dir) / run_name / f"step_{step}"
        save_checkpoint(ckpt_path, nnx.to_pure_dict(single_state), single_opt_state,
                         {"step": step, "loader_state": train_loader.state_dict()})
        log(step=step, event="checkpoint", path=str(ckpt_path))

    sample_dir = Path(args.sample_dir) / run_name
    epoch = 0
    pbar = tqdm(range(start_step, args.steps), initial=start_step, total=args.steps)
    for step in pbar:
        t0 = time.time()
        batch = _prefetch_q.get()
        loss, bpb, grads = p_grad_accum_step(state, batch)
        state, opt_state = p_apply_step(state, opt_state, grads)
        dt_ms = (time.time() - t0) * 1000
        loss0, bpb0 = float(loss[0]), float(bpb[0])
        lr_now = float(lr_schedule(step))
        pbar.set_postfix(loss=loss0, bpb=bpb0, dt_ms=f"{dt_ms:.1f}", lr=f"{lr_now:.2e}")
        log(console=False, step=step, split="train", loss=loss0, bpb=bpb0, dt_ms=dt_ms, lr=lr_now)

        if (step + 1) % args.steps_per_epoch == 0:
            epoch += 1
            single_state = unreplicate(state)
            eval_model = nnx.merge(graphdef, single_state)
            eval_metrics = eval_and_sample(
                eval_model, p_eval_step, eval_step_single, state, single_state, val_loader,
                args.resolution, batch_size, n_devices, args.eval_samples, args.eval_prompt_len,
                args.eval_batches, sample_dir, tag=f"epoch{epoch}_step{step+1}", seed=args.seed + epoch,
            )
            log(step=step + 1, split="eval", epoch=epoch, **eval_metrics,
                samples=[str(sample_dir / f"epoch{epoch}_step{step+1}_{i}.png") for i in range(args.eval_samples)])

        if args.save_every is not None and (step + 1) % args.save_every == 0:
            save(step + 1)

    if args.save_every is not None and args.steps % args.save_every != 0:
        save(args.steps)
    log(event="done", step=args.steps)


if __name__ == "__main__":
    main()
