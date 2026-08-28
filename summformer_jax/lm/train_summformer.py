"""JAX/pmap training loop for summformer_jax -- an ablation of the Ks-hierarchical-summarization
+ fuse-cross-attn method against the gpt2_jax medium baseline. Deliberately mirrors
gpt2_jax/train_gpt.py's conventions exactly (same AdamW hyperparameters/schedule,
total_batch_size-via-grad-accum scheme, fused grad-accum/eval scan, background prefetch,
donate_argnums, tqdm, qcute-style --config parsing) and the SAME already-prepped
data/fineweb-edu-10B GPT2-BPE token shards/vocab/embed (Config.vocab_size=50304 by default) so
the only real variable between the two runs is the architecture itself: gpt2_jax's plain 24-layer
GPT2 block stack vs. this file's n_layers=2 + Ks=(2,)*6 hierarchical cascade at the same
d_model=1024/n_heads=16.

    uv run python summformer_jax/train_summformer.py --pos-method rope \
      --dataset-dir data/fineweb-edu-10B --Ks 2,2,2,2,2 --d-model 1024 --n-heads 16 \
      --n-layers 2
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from checkpoint_io import save_checkpoint
from data_loader import DataLoaderLite
from model_summformer import Config, SummTransformer

TOTAL_BATCH_SIZE_DEFAULT = 2**19  # 524,288 tokens -- same default as gpt2_jax
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 715
NUM_DATASET_TOKENS = 10_000_000_000  # fineweb-edu-10B, same GPT2-BPE token count as gpt2_jax


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
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="summformer_jax training", parents=[pre])
    p.add_argument("--Ks", type=str, default="2,2,2")
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--fuse-n-layers", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--mlp-mult", type=int, default=4)
    p.add_argument("--pos-method", choices=["rope", "learnable", "base"], default="rope")
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--attn-window", type=int, default=-1, help="-1 = max (unbounded)")
    p.add_argument("--fuse-window", type=int, default=-1, help="-1 = max (unbounded)")
    p.add_argument("--input-preset", type=int, default=8, help="byte-alphabet fallback, only used if --vocab-size is 0/unset")
    p.add_argument("--vocab-size", type=int, default=50304, help="0 = fall back to the byte alphabet (2**input_preset)")
    p.add_argument("--mtp-heads", type=int, default=1)
    p.add_argument("--mtp-weight", type=float, default=1.0)
    p.add_argument("--weight-tie", action="store_true")
    p.add_argument("--share-lm", action="store_true")
    p.add_argument("--share-fuse", action="store_true")

    p.add_argument("--dataset-dir", type=str, default=None,
                    help="dir of train_*.npy/val_*.npy GPT2-BPE token shards from gpt2_jax/dataset_preparation.py "
                         "(e.g. data/fineweb-edu-10B) -- same dataset as the gpt2_jax baseline for this ablation")
    p.add_argument("--save-dir", type=str, default="summformer_jax/logs")
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--total-batch-size", type=int, default=TOTAL_BATCH_SIZE_DEFAULT)
    p.add_argument("--batch-size", type=int, default=8, help="per-device micro batch size")
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--run-name", type=str, default=None)

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

    Ks = tuple(int(x) for x in args.Ks.split(","))
    attn_window = None if args.attn_window is None or args.attn_window < 0 else args.attn_window
    fuse_window = None if args.fuse_window is None or args.fuse_window < 0 else args.fuse_window

    n_devices = jax.local_device_count()
    B = args.batch_size
    T = args.sequence_length
    total_batch_size = args.total_batch_size
    assert total_batch_size % (B * T * n_devices) == 0, "total_batch_size must be divisible by batch_size * sequence_length * n_devices"
    grad_accum_steps = total_batch_size // (B * T * n_devices)
    max_steps = args.max_steps or args.num_epochs * (NUM_DATASET_TOKENS // total_batch_size)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"summformer_Ks{args.Ks}_{args.pos_method}")
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
          f"total_batch_size={total_batch_size}  max_steps={max_steps}  pos_method={args.pos_method}  Ks={Ks}")

    train_loader = DataLoaderLite(B, T, 0, 1, "train", args.dataset_dir)
    val_loader = DataLoaderLite(B, T, 0, 1, "val", args.dataset_dir)

    cfg = Config(
        Ks=Ks, d_model=args.d_model, n_layers=args.n_layers, fuse_n_layers=args.fuse_n_layers,
        n_heads=args.n_heads, mlp_mult=args.mlp_mult, pos_method=args.pos_method, rope_base=args.rope_base,
        context_len=T, attn_window=attn_window, fuse_window=fuse_window, input_preset=args.input_preset,
        vocab_size=(args.vocab_size or None),
        mtp_heads=args.mtp_heads, mtp_weight=args.mtp_weight, weight_tie=args.weight_tie,
        share_lm=args.share_lm, share_fuse=args.share_fuse,
    )
    rngs = nnx.Rngs(args.seed)
    model = SummTransformer(cfg, rngs=rngs)
    graphdef, params = nnx.split(model)
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
        # SummTransformer.__call__ takes token_ids (not x/y separately) -- targets are y=x shifted
        # by 1, already encoded in _cascade's internal slicing, so feed x (the B*T window) and
        # let the model's own shift-by-1 produce the same (x[:-1]->x[1:]) pairing as batch's y.
        del y
        loss, metrics = m(x)
        return loss, metrics

    def grad_accum_step(params, batch):
        x_all, y_all = batch  # per-device: [grad_accum_steps, B, T]

        def scan_fn(carry, xy):
            loss_sum, grads_sum, metrics_sum = carry
            x, y = xy
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, (x, y))
            loss_sum = loss_sum + loss
            grads_sum = jax.tree.map(jnp.add, grads_sum, grads)
            metrics_sum = jax.tree.map(jnp.add, metrics_sum, metrics)
            return (loss_sum, grads_sum, metrics_sum), None

        zero_grads = jax.tree.map(jnp.zeros_like, params)
        dummy_loss, dummy_metrics = jax.eval_shape(loss_fn, params, (x_all[0], y_all[0]))
        zero_metrics = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), dummy_metrics)
        zero_loss = jnp.zeros((), dtype=jnp.float32)
        (loss_sum, grads_sum, metrics_sum), _ = jax.lax.scan(
            scan_fn, (zero_loss, zero_grads, zero_metrics), (x_all, y_all))
        loss_mean = loss_sum / grad_accum_steps
        grads_mean = jax.tree.map(lambda g: g / grad_accum_steps, grads_sum)
        metrics_mean = jax.tree.map(lambda m: m / grad_accum_steps, metrics_sum)
        loss_mean = jax.lax.pmean(loss_mean, "batch")
        grads_mean = jax.lax.pmean(grads_mean, "batch")
        metrics_mean = jax.lax.pmean(metrics_mean, "batch")
        return loss_mean, grads_mean, metrics_mean

    p_grad_accum_step = jax.pmap(grad_accum_step, axis_name="batch")

    def apply_step(params, opt_state, grads):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    p_apply_step = jax.pmap(apply_step, axis_name="batch", donate_argnums=(0, 1))

    def eval_accum_step(params, batch):
        x_all, y_all = batch

        def scan_fn(loss_sum, xy):
            x, y = xy
            loss, _ = loss_fn(params, (x, y))
            return loss_sum + loss, None

        loss_sum, _ = jax.lax.scan(scan_fn, jnp.zeros((), dtype=jnp.float32), (x_all, y_all))
        loss_mean = loss_sum / x_all.shape[0]
        return jax.lax.pmean(loss_mean, "batch")

    p_eval_accum_step = jax.pmap(eval_accum_step, axis_name="batch")

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
            log(step=step, split="val", loss=val_loss, bpb=val_loss / math.log(2),
                ppl=math.exp(min(val_loss, 20)))

        batch = _prefetch_q.get()
        loss, grads, metrics = p_grad_accum_step(params, batch)
        params, opt_state = p_apply_step(params, opt_state, grads)
        loss_accum = float(loss[0])

        dt = time.time() - t0
        tok_per_sec = (B * T * grad_accum_steps * n_devices) / dt
        lr = float(schedule(step))
        log(step=step, split="train", loss=loss_accum, bpb=loss_accum / math.log(2),
            ppl=math.exp(min(loss_accum, 20)), lr=lr, dt_ms=dt * 1000, tok_per_sec=tok_per_sec)
        postfix = {"loss": f"{loss_accum:.6f}", "bpb": f"{loss_accum/math.log(2):.6f}",
                   "tok/s": f"{tok_per_sec:.2f}", "lr": f"{lr:.8f}"}
        if val_loss is not None:
            postfix["val_loss"] = f"{val_loss:.6f}"
        pbar.set_postfix(postfix)
        if step == 0:
            print(f"[compile] step 0 wall time (includes first-call XLA compile): {dt:.2f}s", flush=True)

        if last_step:
            single = jax.tree.map(lambda x: x[0], params)
            ckpt_path = log_dir / f"model_{step}"
            save_checkpoint(ckpt_path, single)
            log(step=step, event="checkpoint", path=str(ckpt_path.resolve()))


if __name__ == "__main__":
    main()
