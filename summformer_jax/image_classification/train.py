"""Training loop for image_classification's SummClassifier. Mirrors ref_impl_files/train.py's
(Flax ResNet50 ImageNet example) overall recipe structure -- linear-warmup + cosine-decay LR
schedule, base_lr scaled by batch_size/256 (same linear-scaling-rule convention, see that file's
own `base_learning_rate = config.learning_rate * config.batch_size / 256.0`), ~100-epoch budget,
top1/top5 accuracy logging -- but swaps SGD+momentum for AdamW+weight-decay (ViT/DeiT-style
recipe, not ResNet's), since this is a transformer, not a CNN. jax.pmap data-parallel across all
local devices (image_gen/train.py's own pattern), single micro-batch per device per step (no
grad-accum -- add if a larger effective batch is needed later).

    uv run python summformer_jax/image_classification/train.py --config summformer_jax/image_classification/configs/tiny_vit_like.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from classifier import ClassifierConfig, SummClassifier, cross_entropy_logits, topk_accuracy
from multiscan_classifier import MultiScanConfig, MultiScanClassifier
from dataloader import ImageNetClassificationLoader


def load_config_module(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


def build_config(config_vars: dict):
    """A config file setting `scan_groups` builds a MultiScanConfig (multiple pixel-traversal
    orders, weight-sharing groups, see multiscan_classifier.py); otherwise a plain
    ClassifierConfig (unidirectional or forward+reverse dual-pass, see classifier.py)."""
    if "scan_groups" in config_vars:
        valid = {f.name for f in fields(MultiScanConfig)}
        return MultiScanConfig(**{k: v for k, v in config_vars.items() if k in valid})
    valid = {f.name for f in fields(ClassifierConfig)}
    return ClassifierConfig(**{k: v for k, v in config_vars.items() if k in valid})


def build_model(cfg, *, rngs: nnx.Rngs):
    if isinstance(cfg, MultiScanConfig):
        return MultiScanClassifier(cfg, rngs=rngs)
    return SummClassifier(cfg, rngs=rngs)


def replicate(pytree, devices):
    mesh = jax.sharding.Mesh(devices, ("d",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
    stacked = jax.tree.map(lambda x: jnp.stack([x] * len(devices)), pytree)
    return jax.tree.map(lambda x: jax.device_put(x, sharding), stacked)


def unreplicate(pytree):
    return jax.tree.map(lambda x: x[0], pytree)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--shard-dir", type=str, required=True, help="dir of download_imagenet.py's labeled .bin shards")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=None, help="per-device; overrides config")
    p.add_argument("--num-epochs", type=float, default=None, help="overrides config")
    p.add_argument("--base-lr", type=float, default=None, help="scaled by batch_size/256 (linear scaling rule); overrides config")
    p.add_argument("--warmup-epochs", type=float, default=None, help="overrides config")
    p.add_argument("--weight-decay", type=float, default=None, help="overrides config")
    p.add_argument("--steps-per-epoch-eval", type=int, default=1, help="eval (full val sweep) fires every this many real epochs")
    p.add_argument("--eval-batches", type=int, default=None, help="cap the val sweep to this many batches")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="summformer_jax/image_classification/logs")
    args = p.parse_args()

    config_vars = load_config_module(args.config)
    cfg = build_config(config_vars)

    batch_size = args.batch_size or config_vars.get("batch_size", 32)
    num_epochs = args.num_epochs or config_vars.get("num_epochs", 100.0)
    base_lr = args.base_lr or config_vars.get("base_lr", 5e-4)  # DeiT's own base_lr @ batch=512
    warmup_epochs = args.warmup_epochs or config_vars.get("warmup_epochs", 5.0)
    weight_decay = args.weight_decay or config_vars.get("weight_decay", 0.05)

    run_name = args.run_name or args.config.stem
    log_dir = Path(args.save_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(log_dir / "log.jsonl", "a")

    def log(*, console=True, **record):
        record["t"] = time.time()
        if console:
            tqdm.write(str(record))
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    devices = jax.local_devices()
    n_devices = len(devices)

    train_loader = ImageNetClassificationLoader(batch_size, args.image_size, args.shard_dir, "train", seed=args.seed)
    val_loader = ImageNetClassificationLoader(batch_size, args.image_size, args.shard_dir, "validation", seed=args.seed + 1)
    assert train_loader.seq_len == cfg.context_len, (
        f"loader seq_len={train_loader.seq_len} (image_size={args.image_size}) != cfg.context_len={cfg.context_len}"
    )

    steps_per_epoch = train_loader.n_images // (batch_size * n_devices)
    total_steps = int(num_epochs * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    # DeiT's own linear-scaling convention (base_lr * batch/512), not the Flax ResNet50
    # reference's /256 -- this is a ViT-family recipe (AdamW, weight decay), not ResNet's (SGD)
    peak_lr = base_lr * (batch_size * n_devices) / 512.0

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr, warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1), end_value=peak_lr * 0.01,
    )
    optimizer = optax.adamw(lr_schedule, b1=0.9, b2=0.999, weight_decay=weight_decay)

    model = build_model(cfg, rngs=nnx.Rngs(args.seed))
    graphdef, state = nnx.split(model)
    opt_state = optimizer.init(state)

    def loss_fn(state, tokens, labels):
        model = nnx.merge(graphdef, state)
        logits = model(tokens)
        loss = cross_entropy_logits(logits, labels)
        top1 = topk_accuracy(logits, labels, 1)
        top5 = topk_accuracy(logits, labels, 5)
        return loss, (top1, top5)

    def train_step(state, opt_state, tokens, labels):
        (loss, (top1, top5)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state, tokens, labels)
        loss, top1, top5, grads = (jax.lax.pmean(x, "d") for x in (loss, top1, top5, grads))
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, loss, top1, top5

    p_train_step = jax.pmap(train_step, axis_name="d", donate_argnums=(0, 1))

    def eval_step(state, tokens, labels):
        loss, (top1, top5) = loss_fn(state, tokens, labels)
        return (jax.lax.pmean(x, "d") for x in (loss, top1, top5))

    p_eval_step = jax.pmap(eval_step, axis_name="d")

    state = replicate(state, devices)
    opt_state = replicate(opt_state, devices)

    print(f"config: {config_vars}")
    print(f"n_train_images={train_loader.n_images} n_val_images={val_loader.n_images} "
          f"batch_size(per-device)={batch_size} n_devices={n_devices} steps_per_epoch={steps_per_epoch} "
          f"total_steps={total_steps} peak_lr={peak_lr:.2e} warmup_steps={warmup_steps} run_name={run_name}")

    def fetch_pmap_batch(loader: ImageNetClassificationLoader):
        xs, ys = [], []
        for _ in range(n_devices):
            x, y = loader.next_batch()
            xs.append(x)
            ys.append(y)
        return jnp.asarray(np.stack(xs), dtype=jnp.int32), jnp.asarray(np.stack(ys), dtype=jnp.int32)

    def run_full_eval(state, eval_batches):
        losses, top1s, top5s = [], [], []
        n = 0
        for batch_group in tqdm(_grouped(val_loader.full_sweep(batch_size), n_devices), desc="val sweep", leave=False):
            if eval_batches is not None and n >= eval_batches:
                break
            xs = jnp.asarray(np.stack([b[0] for b in batch_group]), dtype=jnp.int32)
            ys = jnp.asarray(np.stack([b[1] for b in batch_group]), dtype=jnp.int32)
            loss, top1, top5 = p_eval_step(state, xs, ys)
            losses.append(float(loss[0]))
            top1s.append(float(top1[0]))
            top5s.append(float(top5[0]))
            n += 1
        return {"val_loss": sum(losses) / len(losses), "val_top1": sum(top1s) / len(top1s), "val_top5": sum(top5s) / len(top5s)}

    epoch = 0
    pbar = tqdm(range(total_steps))
    for step in pbar:
        tokens, labels = fetch_pmap_batch(train_loader)
        t0 = time.time()
        state, opt_state, loss, top1, top5 = p_train_step(state, opt_state, tokens, labels)
        dt_ms = (time.time() - t0) * 1000
        loss0, top1_0, top5_0 = float(loss[0]), float(top1[0]), float(top5[0])
        lr_now = float(lr_schedule(step))
        pbar.set_postfix(loss=loss0, top1=top1_0, top5=top5_0, dt_ms=f"{dt_ms:.1f}", lr=f"{lr_now:.2e}")
        log(console=False, step=step, split="train", loss=loss0, top1_acc=top1_0, top5_acc=top5_0, dt_ms=dt_ms, lr=lr_now)

        if (step + 1) % (args.steps_per_epoch_eval * steps_per_epoch) == 0:
            epoch += 1
            eval_metrics = run_full_eval(state, args.eval_batches)
            log(step=step + 1, split="eval", epoch=epoch, **eval_metrics)

    log(event="done", step=total_steps)


def _grouped(iterable, n):
    group = []
    for item in iterable:
        group.append(item)
        if len(group) == n:
            yield group
            group = []
    if group:
        # pad the final partial group by repeating the last item (dropped from the mean via
        # eval_batches bookkeeping isn't done here -- acceptable small bias on the very last group
        # of an eval sweep, matches the same tradeoff image_gen/train.py's eval loop accepts)
        while len(group) < n:
            group.append(group[-1])
        yield group


if __name__ == "__main__":
    main()
