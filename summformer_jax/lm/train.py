"""Training loop for the new-architecture (summformer.py) lm lineage -- pmap data-parallel across
all local devices, config file exposes a `build_summformer(rngs) -> ARHead` factory (see
configs/thin512_win16_allfuse8_chained.py) instead of the old flat ConfigV2 dict, since the new
architecture composes Embedder/Encoder/Decoder objects rather than a single dataclass.

    uv run python summformer_jax/lm/train.py --config configs/thin512_win16_allfuse8_chained.py --data-dir data/fineweb-edu-10B
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


def load_config_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShardedTokenLoader:
    """Reads pre-tokenized .npy/.bin uint16 GPT-2 BPE shards (gpt2_jax/dataset_preparation.py's
    output format -- reused directly, same tokenizer/vocab). Falls back to a tiny synthetic corpus
    if no shards are found at data_dir, so this script is runnable without real data present."""
    def __init__(self, data_dir: str, seq_len: int, split: str, seed: int = 0):
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        d = Path(data_dir)
        shard_glob = f"*{split}*.npy" if split != "train" else "*train*.npy"
        self.shards = sorted(d.glob(shard_glob)) if d.exists() else []
        if not self.shards:
            print(f"[ShardedTokenLoader] no shards found at {data_dir} (split={split}) -- using synthetic random tokens")
            self.tokens = None
        else:
            self.tokens = np.concatenate([np.load(s).astype(np.int32) for s in self.shards])
            print(f"[ShardedTokenLoader] loaded {len(self.shards)} shard(s), {len(self.tokens)} tokens (split={split})")

    def next_batch(self, batch_size: int) -> np.ndarray:
        if self.tokens is None:
            return self.rng.integers(0, 50257, size=(batch_size, self.seq_len + 1), dtype=np.int32)
        starts = self.rng.integers(0, len(self.tokens) - self.seq_len - 1, size=batch_size)
        return np.stack([self.tokens[s:s + self.seq_len + 1] for s in starts])


def replicate(pytree, devices):
    mesh = jax.sharding.Mesh(devices, ("d",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
    stacked = jax.tree.map(lambda x: jnp.stack([x] * len(devices)), pytree)
    return jax.tree.map(lambda x: jax.device_put(x, sharding), stacked)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=str, default="data/fineweb-edu-10B")
    p.add_argument("--batch-size", type=int, default=None, help="per-device; overrides config")
    p.add_argument("--total-steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="summformer_jax/lm/logs")
    args = p.parse_args()

    cfg_mod = load_config_module(args.config)
    batch_size = args.batch_size or getattr(cfg_mod, "batch_size", 4)
    seq_len = cfg_mod.SEQUENCE_LENGTH

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

    train_loader = ShardedTokenLoader(args.data_dir, seq_len, "train", seed=args.seed)
    val_loader = ShardedTokenLoader(args.data_dir, seq_len, "val", seed=args.seed + 1)

    head = cfg_mod.build_summformer(nnx.Rngs(args.seed))
    graphdef, state = nnx.split(head)
    optimizer = optax.adamw(args.lr, b1=0.9, b2=0.95, weight_decay=0.1)
    opt_state = optimizer.init(state)

    def loss_fn(state, batch):
        m = nnx.merge(graphdef, state)
        loss, metrics = m(batch)
        return loss, metrics

    def train_step(state, opt_state, batch):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state, batch)
        loss, metrics, grads = (jax.lax.pmean(x, "d") for x in (loss, metrics, grads))
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, loss, metrics

    p_train_step = jax.pmap(train_step, axis_name="d", donate_argnums=(0, 1))

    def eval_step(state, batch):
        loss, metrics = loss_fn(state, batch)
        return jax.lax.pmean(loss, "d"), jax.lax.pmean(metrics["bpb"], "d")

    p_eval_step = jax.pmap(eval_step, axis_name="d")

    state = replicate(state, devices)
    opt_state = replicate(opt_state, devices)

    n_params = sum(x.size for x in jax.tree.leaves(nnx.state(head, nnx.Param)))
    print(f"run_name={run_name} n_devices={n_devices} batch_size(per-device)={batch_size} "
          f"seq_len={seq_len} n_params={n_params} ({n_params/1e6:.1f}M) total_steps={args.total_steps}")

    def fetch_pmap_batch(loader):
        return jnp.asarray(np.stack([loader.next_batch(batch_size) for _ in range(n_devices)]), dtype=jnp.int32)

    pbar = tqdm(range(args.total_steps))
    for step in pbar:
        batch = fetch_pmap_batch(train_loader)
        t0 = time.time()
        state, opt_state, loss, metrics = p_train_step(state, opt_state, batch)
        dt_ms = (time.time() - t0) * 1000
        loss0, bpb0 = float(loss[0]), float(metrics["bpb"][0])
        pbar.set_postfix(loss=loss0, bpb=bpb0, dt_ms=f"{dt_ms:.1f}")
        log(console=False, step=step, split="train", loss=loss0, bpb=bpb0, dt_ms=dt_ms)

        if (step + 1) % args.eval_every == 0:
            val_batch = fetch_pmap_batch(val_loader)
            val_loss, val_bpb = p_eval_step(state, val_batch)
            log(step=step + 1, split="eval", val_loss=float(val_loss[0]), val_bpb=float(val_bpb[0]))

    log(event="done", step=args.total_steps)


if __name__ == "__main__":
    main()
