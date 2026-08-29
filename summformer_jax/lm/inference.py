"""Inference + consistency checks for lm/summformer.py, on a random-init model (no checkpoint
needed) -- ported from summformer_jax/image_gen/inference.py, adapted for text (GPT2-BPE via
tiktoken, or raw bytes when vocab_size is unset) instead of RGB image bytes. Same three checks,
no PNG output:

  1. generate_no_cache produces a valid token sequence (0 <= id < vocab_size) of the requested length.
  2. Two calls with the same key produce the SAME output (determinism), two different keys
     produce DIFFERENT output (sampling actually varies) -- basic sanity, not architecture-specific.
  3. check_block_locality: probes whether a distant token affects a given position's logits,
     reported alongside whether that position is inside the token's main_window or reachable via a
     fuse-stage, so you can visually confirm the block-local-plus-reconnection structure matches
     what the config claims.

    uv run python summformer_jax/lm/inference.py --config configs/thin512_win2_allfuse12.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from flax import nnx

from summformer import ConfigV2, SummTransformerV2, check_block_locality
from train import load_config_module


def build_config(config_vars: dict) -> ConfigV2:
    """Same field mapping train.py's main() applies via argparse defaults, minus everything
    training-specific (dataset_dir, batch sizes, lr, ...) -- inference only needs the architecture."""
    return ConfigV2(
        n_layers=config_vars.get("n_layers", 12),
        d_model=config_vars.get("d_model", 1024),
        n_heads=config_vars.get("n_heads", 16),
        mlp_mult=config_vars.get("mlp_mult", 4),
        pos_method=config_vars.get("pos_method", "rope"),
        rope_base=config_vars.get("rope_base", 10000.0),
        context_len=config_vars.get("sequence_length", 1024),
        main_window=config_vars.get("main_window", -1),
        fuse_stages=tuple(tuple(s) for s in config_vars.get("fuse_stages", ())),
        input_preset=config_vars.get("input_preset", 8),
        vocab_size=config_vars.get("vocab_size", 50304) or None,
        mtp_heads=config_vars.get("mtp_heads", 1),
        mtp_weight=config_vars.get("mtp_weight", 1.0),
        weight_tie=config_vars.get("weight_tie", False),
        zero_kv_sink=config_vars.get("zero_kv_sink", True),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--prompt", type=str, default=None, help="text prompt; default: random tokens")
    p.add_argument("--prompt-len", type=int, default=32, help="tokens of random prompt when --prompt is unset")
    p.add_argument("--gen-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config_vars = load_config_module(args.config)
    cfg = build_config(config_vars)
    is_bpe = cfg.vocab_size is not None and cfg.vocab_size > 256

    model = SummTransformerV2(cfg, rngs=nnx.Rngs(args.seed))
    V = model.vocab

    enc = None
    if is_bpe:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

    if args.prompt is not None:
        ids = enc.encode_ordinary(args.prompt) if enc is not None else list(args.prompt.encode("utf-8"))
        prompt = jnp.array(ids, dtype=jnp.int32)
    else:
        key = jax.random.PRNGKey(args.seed)
        prompt = jax.random.randint(key, (args.prompt_len,), 0, V)

    # --- check 1: valid token sequence ---
    out = model.generate_no_cache(prompt, args.gen_tokens, key=None)  # greedy
    assert out.shape[0] == prompt.shape[0] + args.gen_tokens
    assert bool(jnp.all((out >= 0) & (out < V)))
    print(f"[PASS] generate_no_cache produced {out.shape[0]} valid tokens (greedy)")

    # --- check 2: determinism + sampling variation ---
    out_a = model.generate_no_cache(prompt, 16, key=jax.random.PRNGKey(42), temperature=1.0)
    out_b = model.generate_no_cache(prompt, 16, key=jax.random.PRNGKey(42), temperature=1.0)
    out_c = model.generate_no_cache(prompt, 16, key=jax.random.PRNGKey(43), temperature=1.0)
    same_key_match = bool(jnp.array_equal(out_a, out_b))
    diff_key_differs = not bool(jnp.array_equal(out_a, out_c))
    print(f"[{'PASS' if same_key_match else 'FAIL'}] same key -> identical output")
    print(f"[{'PASS' if diff_key_differs else 'FAIL'}] different key -> different output")

    # --- check 3: block-locality probe ---
    seq_len = cfg.context_len
    print("\nblock-locality probe (query_pos, probe_pos, affected, max_abs_diff):")
    window = cfg.main_window if isinstance(cfg.main_window, int) and cfg.main_window != -1 else None
    probes = [
        ("probe just inside window", seq_len // 2, seq_len // 2 - (window - 1 if window else 1)),
        ("probe just outside window, no fuse-stage reach", seq_len // 2, max(0, seq_len // 2 - (window + 5 if window else seq_len))),
        ("probe far away (start of sequence)", seq_len - 1, 0),
    ]
    for label, q, pr in probes:
        if q >= seq_len or pr >= q:
            continue
        result = check_block_locality(model, nnx.Rngs(args.seed), seq_len, q, pr)
        print(f"  {label}: q={q} probe={pr} affected={result['affected']} max_abs_diff={result['max_abs_diff']:.6f}")

    # --- decode a sample of the greedy continuation ---
    if enc is not None:
        text = enc.decode(list(int(t) for t in out))
    else:
        text = bytes(int(t) for t in out).decode("utf-8", errors="replace")
    print(f"\ngenerated (greedy): {text!r}")


if __name__ == "__main__":
    main()
