"""Wallclock prefill + generation latency benchmark for gpt2_jax or summformer_jax, on TPU or CPU.
Random weights (freshly nnx.Rngs-initialized, no checkpoint) -- this is a speed benchmark only, not
a correctness/quality one. Single device (no pmap) -- one replica's latency, not multi-chip
throughput. Standard warmup + repeated-trial methodology: `--warmup` untimed iterations to let XLA
compile settle, then `--repeats` timed iterations, reporting mean/std/min/max wallclock.

Prefill = one forward pass over the full context (length from --config's context_len/block_size,
override with --context). Generation = `--gen-tokens` further tokens one at a time after prefill
(default 1), via each model's real incremental KV-cache stepper (O(1) work/token) -- both
summformer_jax and gpt2_jax have one (gpt2_jax's added 2026-08-28, mirroring summformer's). Pass
--naive-generation to instead use gpt2's old full-context-recompute-per-token path (no cache), kept
for comparison against the pre-cache numbers.

Both prefill and (prefill+generation) are wrapped in a SINGLE `jax.jit` per shape and called
repeatedly, rather than left as eager per-op dispatch. This matters because `--gen-tokens` is a
fixed Python int per run, so the whole prefill-then-generate-N-more trajectory has statically
known shapes throughout -- the Python for-loop over generation steps unrolls fully at trace time,
producing one compiled program. Critically, this is NOT the same as jitting the incremental
stepper's mutating closure and calling that jitted function repeatedly across separate Python
calls -- that would silently freeze the KV cache at whatever it was on the FIRST trace, since a
jit replay reuses the compiled executable without re-running the Python-level mutation. Instead
each timed call builds a fresh stepper/cache from scratch and runs the entire fixed-length
trajectory in one self-contained trace, so cache growth across generation steps is correct WITHIN
a call, and every call after the first hits the same compiled executable (fair "compiled steady
state" timing, not per-op dispatch overhead). generation_only cost is then reported as
(combined mean) - (prefill-alone mean), amortized over --gen-tokens.

    uv run python scripts/jax/bench_generation.py --model summformer --config configs/summformer_jax/medium_rope_ablation.py --device tpu
    uv run python scripts/jax/bench_generation.py --model gpt2 --config configs/gpt2_jax/medium_rope_default.py --device tpu --context 512 --gen-tokens 32
    uv run python scripts/jax/bench_generation.py --model summformer --config configs/summformer_jax/small_rope_ablation.py   # --device cpu default
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "gpt2_jax"))
sys.path.insert(0, str(REPO_ROOT / "summformer_jax" / "lm"))


def load_config_module(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


def timed_runs(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return times


def summarize(times: list[float]) -> dict:
    return {
        "mean_s": statistics.mean(times),
        "std_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_s": min(times),
        "max_s": max(times),
        "all_s": times,
    }


def build_gpt2(cfg_vars: dict, context: int | None, seed: int, flash_override: bool | None):
    from train_gpt import MODEL_SHAPES
    from model_gpt import Model, ModelConfig

    model_name = cfg_vars.get("model", "tiny")
    pos_method = cfg_vars.get("pos_method", "rope")
    T = context or cfg_vars.get("sequence_length", 1024)
    use_flash = cfg_vars.get("use_flash_attention", False) if flash_override is None else flash_override
    mcfg = ModelConfig(pos_method=pos_method, block_size=T, vocab_size=50304,
                        use_flash_attention=use_flash, **MODEL_SHAPES[model_name])
    model = Model(mcfg, rngs=nnx.Rngs(seed))
    return model, T, {"model": model_name, "pos_method": pos_method, "use_flash_attention": use_flash}


def build_summformer(cfg_vars: dict, context: int | None, seed: int, zero_kv_sink: bool):
    from model_summformer import Config, SummTransformer

    Ks_raw = cfg_vars.get("Ks", "2,2,2")
    Ks = tuple(int(x) for x in Ks_raw.split(",")) if isinstance(Ks_raw, str) else tuple(Ks_raw)
    T = context or cfg_vars.get("context_len", 1024)
    scfg = Config(
        Ks=Ks, d_model=cfg_vars.get("d_model", 1024), n_heads=cfg_vars.get("n_heads", 16),
        n_layers=cfg_vars.get("n_layers", 2), pos_method=cfg_vars.get("pos_method", "rope"),
        context_len=T, vocab_size=cfg_vars.get("vocab_size", 50304), zero_kv_sink=zero_kv_sink,
    )
    model = SummTransformer(scfg, rngs=nnx.Rngs(seed))
    return model, T, {"Ks": Ks, "d_model": scfg.d_model, "n_heads": scfg.n_heads,
                       "zero_kv_sink": zero_kv_sink,
                       "n_layers": scfg.n_layers, "pos_method": scfg.pos_method}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["gpt2", "summformer"], required=True)
    ap.add_argument("--config", type=Path, default=None, help="Python config file, same format train_*.py takes")
    ap.add_argument("--context", type=int, default=None, help="prefill length; default from --config's context_len/sequence_length (else 1024)")
    ap.add_argument("--gen-tokens", type=int, default=1, help="tokens to generate one-at-a-time after prefill")
    ap.add_argument("--device", choices=["cpu", "tpu"], default="cpu", help="single device only, no pmap -- benchmark focuses on TPU, cpu is the portable default")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3, help="untimed iterations before measuring, standard practice to let XLA compile settle")
    ap.add_argument("--repeats", type=int, default=10, help="timed iterations to average over")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=50304)
    ap.add_argument("--flash-attention", dest="flash_attention", action="store_true", default=None,
                     help="force gpt2's flash-attention on, overriding --config")
    ap.add_argument("--no-flash-attention", dest="flash_attention", action="store_false",
                     help="force it off -- for a fair comparison against summformer, which has no "
                          "flash-attention path at all (see docs/status_tpu.md's note on why "
                          "adding one there is nontrivial: zero-KV-sink + flash-attention was "
                          "found ~25x slower in the bytelm_tpu lineage)")
    ap.add_argument("--zero-kv-sink", dest="zero_kv_sink", action="store_true", default=True,
                     help="summformer only: use its zero-KV-sink attention (default, matches all "
                          "trained checkpoints -- purely a runtime toggle, no learned params, so "
                          "flipping it doesn't touch checkpoint compatibility)")
    ap.add_argument("--no-zero-kv-sink", dest="zero_kv_sink", action="store_false",
                     help="isolate the sink's own cost -- for bench comparisons only, not for "
                          "evaluating a real checkpoint (all of which were trained with it on)")
    ap.add_argument("--naive-generation", action="store_true",
                     help="gpt2 only: use the old naive full-recompute-per-token path instead of "
                          "the real incremental KV-cache stepper (added 2026-08-28) -- for "
                          "comparing against the pre-cache numbers, not the default")
    ap.add_argument("--out", type=Path, default=None, help="default: bench_results/<model>_<config-or-manual>_<device>_<timestamp>.json")
    args = ap.parse_args()

    devices = jax.devices(args.device)
    assert devices, f"no {args.device} devices visible"
    device = devices[0]
    print(f"device: {device} (of {len(devices)} {args.device} device(s) visible, using 1)")

    cfg_vars = load_config_module(args.config) if args.config else {}

    with jax.default_device(device):
        if args.model == "gpt2":
            model, T, model_desc = build_gpt2(cfg_vars, args.context, args.seed, args.flash_attention)
        else:
            model, T, model_desc = build_summformer(cfg_vars, args.context, args.seed, args.zero_kv_sink)

        graphdef, state = nnx.split(model)
        n_params = sum(x.size for x in jax.tree.leaves(state) if hasattr(x, "size"))
        print(f"model={args.model}  context={T}  gen_tokens={args.gen_tokens}  batch={args.batch}  "
              f"params={n_params:,}  {model_desc}")

        key = jax.random.PRNGKey(args.seed)
        prompt = jax.random.randint(key, (args.batch, T), 0, args.vocab)
        gen_tokens = args.gen_tokens  # fixed Python int -- unrolled at trace time, see docstring

        # --- pure, jittable prefill/run functions ---
        # `state` is passed explicitly (not closed over) so jax.jit treats it as a real input,
        # matching the nnx.split/merge pattern train_gpt.py/train_summformer.py already use for
        # their own jitted train steps.
        if args.model == "gpt2":
            def prefill_pure(state, prompt):
                m = nnx.merge(graphdef, state)
                return m(prompt)

            if args.naive_generation:
                def run_pure(state, prompt, key):
                    # gpt2_jax's flash-attention kernel requires kv_seq_len % 128 == 0 at every
                    # call; naive per-token growth breaks that on every step except when the grown
                    # length happens to land on a multiple of 128. Pad each step's forward pass out
                    # to the next 128-boundary with dummy tokens -- causal masking means padding
                    # (appended after all real content) never affects any real position's logits,
                    # so this is correctness-preserving, just satisfies the kernel's requirement.
                    m = nnx.merge(graphdef, state)
                    seq = prompt
                    logits = m(seq)
                    for i in range(gen_tokens):
                        next_tok = jax.random.randint(jax.random.fold_in(key, i), (args.batch, 1), 0, args.vocab)
                        seq = jnp.concatenate([seq, next_tok], axis=1)  # naive: no cache, context grows
                        cur_len = seq.shape[1]
                        padded_len = -(-cur_len // 128) * 128  # ceil to next multiple of 128
                        pad = jnp.zeros((args.batch, padded_len - cur_len), dtype=seq.dtype)
                        logits = m(jnp.concatenate([seq, pad], axis=1))[:, :cur_len, :]
                    return logits
            else:
                def run_pure(state, prompt, key):
                    # real incremental KV-cache (gpt2_jax/model_gpt.py's Model._make_incremental_
                    # stepper, added 2026-08-28) -- always plain SDPA internally, never flash
                    # (that kernel needs kv_seq_len % 128 == 0, incompatible with single-token
                    # growing cache steps), so --flash-attention only affects prefill here.
                    m = nnx.merge(graphdef, state)
                    step = m._make_incremental_stepper(args.batch)
                    logits = step(prompt, 0)
                    pos = T
                    for i in range(gen_tokens):
                        next_tok = jax.random.randint(jax.random.fold_in(key, i), (args.batch, 1), 0, args.vocab)
                        logits = step(next_tok, pos)
                        pos += 1
                    return logits
        else:
            def prefill_pure(state, prompt):
                m = nnx.merge(graphdef, state)
                step = m._make_incremental_stepper(args.batch)
                return step(prompt, 0)

            def run_pure(state, prompt, key):
                m = nnx.merge(graphdef, state)
                step = m._make_incremental_stepper(args.batch)  # fresh cache, this trace only
                logits = step(prompt, 0)
                pos = T
                for i in range(gen_tokens):
                    next_tok = jax.random.randint(jax.random.fold_in(key, i), (args.batch, 1), 0, args.vocab)
                    logits = step(next_tok, pos)  # real incremental KV-cache step
                    pos += 1
                return logits

        prefill_jit = jax.jit(prefill_pure)
        run_jit = jax.jit(run_pure) if gen_tokens > 0 else None

        # --- prefill (isolated) ---
        prefill_fn = lambda: prefill_jit(state, prompt)
        prefill_times = timed_runs(prefill_fn, args.warmup, args.repeats)
        prefill_stats = summarize(prefill_times)
        print(f"prefill (context={T}, jitted): mean={prefill_stats['mean_s']*1000:.2f}ms  "
              f"std={prefill_stats['std_s']*1000:.2f}ms  min={prefill_stats['min_s']*1000:.2f}ms  "
              f"({T*args.batch/prefill_stats['mean_s']:.0f} tok/s)")

        # --- prefill+generation (combined, one jit call per repeat), generation cost derived by
        # subtracting the isolated prefill mean measured above ---
        gen_stats = None
        if gen_tokens > 0:
            run_fn = lambda: run_jit(state, prompt, key)
            combined_times = timed_runs(run_fn, args.warmup, args.repeats)
            combined_stats = summarize(combined_times)
            gen_total_mean = combined_stats["mean_s"] - prefill_stats["mean_s"]
            per_token_mean = gen_total_mean / gen_tokens
            gen_stats = {
                "combined_mean_s": combined_stats["mean_s"], "combined_std_s": combined_stats["std_s"],
                "combined_all_s": combined_stats["all_s"],
                "prefill_alone_mean_s": prefill_stats["mean_s"],
                "generation_total_mean_s": gen_total_mean, "per_token_mean_s": per_token_mean,
            }
            gen_mode = "naive full recompute, no cache" if (args.model == "gpt2" and args.naive_generation) else "incremental KV-cache"
            print(f"generation ({gen_tokens} token(s) after prefill, jitted as one combined trace, "
                  f"{gen_mode}): "
                  f"combined_mean={combined_stats['mean_s']*1000:.2f}ms  "
                  f"generation_only={gen_total_mean*1000:.2f}ms total, "
                  f"{per_token_mean*1000:.2f}ms/token ({args.batch/per_token_mean:.1f} tok/s)")

    result = {
        "model": args.model, "config": str(args.config) if args.config else None,
        "model_desc": {k: (list(v) if isinstance(v, tuple) else v) for k, v in model_desc.items()},
        "context": T, "gen_tokens": args.gen_tokens, "batch": args.batch, "n_params": n_params,
        "device": str(device), "device_kind": args.device, "warmup": args.warmup, "repeats": args.repeats,
        "seed": args.seed, "prefill": prefill_stats, "generation": gen_stats, "t": time.time(),
    }

    out_path = args.out
    if out_path is None:
        out_dir = REPO_ROOT / "bench_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_tag = args.config.stem if args.config else "manual"
        out_path = out_dir / f"{args.model}_{cfg_tag}_{args.device}_{int(result['t'])}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
