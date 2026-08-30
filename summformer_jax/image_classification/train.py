"""Training loop for the new-architecture (summformer.py) image_classification lineage -- reuses
dataloader.py's ImageNetClassificationLoader (labeled shard reader) unchanged, only the
model-building side is new: config exposes a `build_summformer(rngs) -> ClassifierHead` factory
(see configs/tiny_vit_like_chained.py) instead of the old flat dataclass config.

    uv run python summformer_jax/image_classification/train.py --config configs/tiny_vit_like_chained.py --shard-dir /dev/shm/imagenet_raw
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # summformer_jax/

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from dataloader import ImageNetClassificationLoader


def load_config_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replicate(pytree, devices):
    mesh = jax.sharding.Mesh(devices, ("d",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
    stacked = jax.tree.map(lambda x: jnp.stack([x] * len(devices)), pytree)
    return jax.tree.map(lambda x: jax.device_put(x, sharding), stacked)


def topk_accuracy(logits, labels, k):
    topk_preds = jax.lax.top_k(logits, k)[1]
    hit = jnp.any(topk_preds == labels[:, None], axis=-1)
    return hit.astype(jnp.float32).mean()


def cross_entropy_logits(logits, labels):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--shard-dir", type=str, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=None, help="per-device; overrides config")
    p.add_argument("--num-epochs", type=float, default=None, help="overrides config; real epoch count, not raw steps")
    p.add_argument("--total-steps", type=int, default=None, help="overrides --num-epochs if given (raw step count, for smoke tests)")
    p.add_argument("--lr", type=float, default=None, help="direct AdamW lr (LM-style, no DeiT batch/512 scaling) -- if unset, falls back to config's base_lr with DeiT scaling")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="summformer_jax/image_classification/logs")
    args = p.parse_args()

    cfg_mod = load_config_module(args.config)
    batch_size = args.batch_size or getattr(cfg_mod, "batch_size", 32)
    base_lr = getattr(cfg_mod, "base_lr", 5e-4)
    weight_decay = getattr(cfg_mod, "weight_decay", 0.05)
    num_epochs = args.num_epochs if args.num_epochs is not None else getattr(cfg_mod, "num_epochs", 100.0)
    warmup_epochs = getattr(cfg_mod, "warmup_epochs", 5.0)

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
    assert train_loader.seq_len == cfg_mod.CONTEXT_LEN, (
        f"loader seq_len={train_loader.seq_len} != cfg.CONTEXT_LEN={cfg_mod.CONTEXT_LEN}")

    head = cfg_mod.build_summformer(nnx.Rngs(args.seed))
    graphdef, state = nnx.split(head)

    steps_per_epoch = train_loader.n_images // (batch_size * n_devices)
    total_steps = args.total_steps if args.total_steps is not None else max(1, int(num_epochs * steps_per_epoch))
    warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))

    if args.lr is not None:
        peak_lr = args.lr  # direct, LM-style -- no DeiT batch/512 scaling
    else:
        peak_lr = base_lr * (batch_size * n_devices) / 512.0
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr, warmup_steps=warmup_steps, decay_steps=max(total_steps, warmup_steps + 1), end_value=peak_lr * 0.01)
    optimizer = optax.adamw(lr_schedule, b1=0.9, b2=0.999, weight_decay=weight_decay)
    opt_state = optimizer.init(state)

    def loss_fn(state, tokens, labels):
        m = nnx.merge(graphdef, state)
        logits = m(tokens)
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
        return jax.lax.pmean(loss, "d"), jax.lax.pmean(top1, "d"), jax.lax.pmean(top5, "d")

    p_eval_step = jax.pmap(eval_step, axis_name="d")

    state = replicate(state, devices)
    opt_state = replicate(opt_state, devices)

    n_params = sum(x.size for x in jax.tree.leaves(nnx.state(head, nnx.Param)))
    print(f"run_name={run_name} n_devices={n_devices} batch_size(per-device)={batch_size} "
          f"context_len={cfg_mod.CONTEXT_LEN} n_params={n_params} ({n_params/1e6:.1f}M) "
          f"total_steps={total_steps} peak_lr={peak_lr:.2e}")

    def fetch_pmap_batch(loader):
        xs, ys = [], []
        for _ in range(n_devices):
            x, y = loader.next_batch()
            xs.append(x)
            ys.append(y)
        return jnp.asarray(np.stack(xs), dtype=jnp.int32), jnp.asarray(np.stack(ys), dtype=jnp.int32)

    def run_eval(state, n_batches):
        losses, top1s, top5s = [], [], []
        for _ in range(n_batches):
            x, y = fetch_pmap_batch(val_loader)
            loss, top1, top5 = p_eval_step(state, x, y)
            losses.append(float(loss[0])); top1s.append(float(top1[0])); top5s.append(float(top5[0]))
        return {"val_loss": sum(losses)/len(losses), "val_top1": sum(top1s)/len(top1s), "val_top5": sum(top5s)/len(top5s)}

    pbar = tqdm(range(total_steps))
    for step in pbar:
        # queue depth BEFORE the fetch that's about to drain it -- a near-empty queue means the
        # background workers aren't keeping ahead of consumption (CPU-bound stall imminent); a
        # consistently-full queue means CPU is comfortably ahead and TPU is the bottleneck (good).
        qsize_before = train_loader._queue.qsize()
        t_fetch0 = time.time()
        tokens, labels = fetch_pmap_batch(train_loader)
        fetch_ms = (time.time() - t_fetch0) * 1000
        t0 = time.time()
        state, opt_state, loss, top1, top5 = p_train_step(state, opt_state, tokens, labels)
        dt_ms = (time.time() - t0) * 1000
        loss0, top1_0, top5_0 = float(loss[0]), float(top1[0]), float(top5[0])
        lr_now = float(lr_schedule(step))
        pbar.set_postfix(loss=loss0, top1=top1_0, dt_ms=f"{dt_ms:.1f}", fetch_ms=f"{fetch_ms:.1f}", q=qsize_before, lr=f"{lr_now:.2e}")
        log(console=False, step=step, split="train", loss=loss0, top1_acc=top1_0, top5_acc=top5_0,
            dt_ms=dt_ms, fetch_ms=fetch_ms, queue_depth=qsize_before, lr=lr_now)

        if (step + 1) % args.eval_every == 0:
            eval_metrics = run_eval(state, args.eval_batches)
            log(step=step + 1, split="eval", **eval_metrics)

    log(event="done", step=total_steps)


if __name__ == "__main__":
    main()
