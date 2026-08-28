"""Single-device training loop for image_gen/summformer.py's ConfigV2/SummTransformerV2, reading
whole raster-RGB-byte images via imagenet_dataloader.py. Deliberately simple (no pmap/grad-accum/
prefetch-thread machinery like summformer_jax/lm/train_summformer_v2.py) -- this is the smoke-test
scale train script for image_gen's block-local design, not a multi-hour TPU run yet.

--shard-dir omitted -> synthetic random-RGB batches (no real dataset needed, per the "random rgb
tensor 64x64x3" testing scope).

Every eval (one per --steps-per-epoch) computes a REAL full-validation-set bpd (bits per dimension
-- identical to bits-per-byte here, since each RGB byte is one dimension; the density-estimation
literature -- PixelRNN/PixelCNN, Fractal Generative Models -- reports bpd, so this script does too)
sweep over the val split, not a single random batch, AND generates+saves sample images. Train/val
split follows the standard downsampled-ImageNet convention (ImageNet-1k's own train/validation
split used directly -- see imagenet_dataloader.py's docstring); --shard-dir must hold both
`*train*.npy` and `*validation*.npy` shards (scripts/prep_imagenet64.py's own output layout).
Sample generation via generate_kv_cache_fully_static (the verified-bit-exact static-cache path,
see docs/image_gen_design.md), seeded from a short random prompt. Generation is still
sequential/slow (no jax.jit wrapping the fuse-stage side yet, same doc) -- --eval-samples defaults
to 1 and --eval-batches bounds the val sweep, to keep eval overhead bounded; raise both for a real
run once speed work lands.

    uv run python summformer_jax/image_gen/train.py --config summformer_jax/image_gen/configs/image64_test.py --steps 20
    uv run python summformer_jax/image_gen/train.py --config summformer_jax/image_gen/configs/image64_test.py --steps 20 --shard-dir data/imagenet64/imagenet64_train
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from dataclasses import fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from PIL import Image

from summformer import ConfigV2, SummTransformerV2
from imagenet_dataloader import ImageByteLoader


def load_config_module(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


def build_config(config_vars: dict) -> ConfigV2:
    valid = {f.name for f in fields(ConfigV2)}
    return ConfigV2(**{k: v for k, v in config_vars.items() if k in valid})


def full_val_bpd(model: SummTransformerV2, val_loader: ImageByteLoader, batch_size: int,
                  eval_batches: int | None = None) -> tuple[float, float]:
    """Real sweep over the val split (or the first eval_batches batches, for speed) -- not one
    random batch. Returns (mean_loss, mean_bpd); bpd == bpb here (one RGB byte = one dimension),
    named bpd to match the density-estimation literature's own reporting convention."""
    losses = []
    for i, batch in enumerate(val_loader.full_sweep(batch_size)):
        if eval_batches is not None and i >= eval_batches:
            break
        tokens = jnp.asarray(batch, dtype=jnp.int32)
        loss, _ = model(tokens)
        losses.append(float(loss))
    mean_loss = sum(losses) / len(losses)
    return mean_loss, mean_loss / math.log(2)


def eval_and_sample(model: SummTransformerV2, val_loader: ImageByteLoader, resolution: int, batch_size: int,
                     n_samples: int, prompt_len: int, eval_batches: int | None,
                     sample_dir: Path, tag: str, seed: int) -> dict:
    """Full-val-set bpd (not one batch), plus n_samples generated images saved to
    sample_dir/<tag>_<i>.png -- each seeded from a REAL val image's first prompt_len bytes
    (completion-style, standard in this literature for qualitative inspection -- a random-byte
    prompt looks like noise, not the start of a coherent image) and drawn via true stochastic
    sampling (temperature=1, not the greedy default -- a qualitative sample should be an actual
    draw from the model's distribution, not the single most-likely, often-repetitive continuation)."""
    val_loss, val_bpd = full_val_bpd(model, val_loader, batch_size, eval_batches)

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--shard-dir", type=str, default=None, help="omit for synthetic random-RGB batches")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=10, help="eval (loss + sample generation) fires every this many steps")
    p.add_argument("--eval-samples", type=int, default=1, help="images to generate per eval -- generation is slow (sequential), keep small")
    p.add_argument("--eval-prompt-len", type=int, default=16, help="bytes of a REAL val image used as the completion prompt for sample generation")
    p.add_argument("--eval-batches", type=int, default=None, help="cap the full-val-set bpd sweep to this many batches (default: sweep the whole val split)")
    p.add_argument("--sample-dir", type=str, default="summformer_jax/image_gen/samples")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default=None, help="defaults to the config file's stem")
    p.add_argument("--save-dir", type=str, default="summformer_jax/image_gen/logs")
    args = p.parse_args()

    config_vars = load_config_module(args.config)
    cfg = build_config(config_vars)
    batch_size = config_vars.get("batch_size", 2)

    run_name = args.run_name or args.config.stem
    log_dir = Path(args.save_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(log_dir / "log.jsonl", "a")

    def log(**record):
        record["t"] = time.time()
        print(record)
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

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
    optimizer = optax.adamw(args.lr, b1=0.9, b2=0.95, weight_decay=0.1)
    opt_state = optimizer.init(state)

    def loss_fn(state, tokens):
        model = nnx.merge(graphdef, state)
        loss, metrics = model(tokens)
        return loss, metrics

    @jax.jit
    def train_step(state, opt_state, tokens):
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(state, tokens)
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, loss, metrics

    print(f"config: {config_vars}")
    print(f"seq_len={train_loader.seq_len} batch_size={batch_size} synthetic={train_loader.synthetic} run_name={run_name}")

    sample_dir = Path(args.sample_dir) / run_name
    epoch = 0
    for step in range(args.steps):
        batch = train_loader.next_batch()
        tokens = jnp.asarray(batch, dtype=jnp.int32)
        t0 = time.time()
        state, opt_state, loss, metrics = train_step(state, opt_state, tokens)
        dt_ms = (time.time() - t0) * 1000
        log(step=step, split="train", loss=float(loss), bpb=float(metrics["bpb"]), dt_ms=dt_ms)

        if (step + 1) % args.steps_per_epoch == 0:
            epoch += 1
            eval_model = nnx.merge(graphdef, state)
            eval_metrics = eval_and_sample(
                eval_model, val_loader, args.resolution, batch_size, args.eval_samples, args.eval_prompt_len,
                args.eval_batches, sample_dir, tag=f"epoch{epoch}_step{step+1}", seed=args.seed + epoch,
            )
            log(step=step + 1, split="eval", epoch=epoch, **eval_metrics,
                samples=[str(sample_dir / f"epoch{epoch}_step{step+1}_{i}.png") for i in range(args.eval_samples)])

    log(event="done", step=args.steps)


if __name__ == "__main__":
    main()
