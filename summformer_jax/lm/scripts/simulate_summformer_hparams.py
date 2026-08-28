"""Sweep summformer_jax `Ks`/`n_layers` combinations and report effective_depth/params/FLOPs/KV
cache for each, alongside a target gpt2_jax baseline -- lets you pick hparams before committing to
an actual multi-hour TPU run. Params/FLOPs/KV cache formulas match compare_summformer_gpt2.py
(same effective_depth definition, same analytical KV-cache formula); FLOPs are measured for real
via JAX's compiled cost analysis, not estimated, so this is slower per row than a pure formula sweep
but trustworthy.

    uv run python summformer_jax/lm/scripts/simulate_summformer_hparams.py --target medium
    uv run python summformer_jax/lm/scripts/simulate_summformer_hparams.py --target small --ks-lengths 2,3,4 \
      --ks-values 2,4 --n-layers 1,2,3
    uv run python summformer_jax/lm/scripts/simulate_summformer_hparams.py --target medium --gpt-n-layer 16
      # hypothetical 16-layer gpt2 reference, d_model/n_heads still from --target's preset
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "gpt2_jax"))
sys.path.insert(0, str(_REPO_ROOT / "summformer_jax" / "lm"))

from model_gpt import ModelConfig as GPTConfig, Model as GPT  # noqa: E402
from model_summformer import Config as SummConfig, SummTransformer, cross_entropy  # noqa: E402

TARGETS = {
    # name: (n_layer, n_head, n_embd)
    "tiny": (6, 8, 512),
    "small": (12, 12, 768),
    "medium": (24, 16, 1024),
}
CONTEXT_LEN = 1024
VOCAB = 50304
DTYPE_BYTES = 2  # bf16


def count_params(module) -> int:
    _, state = nnx.split(module)
    return sum(v.size for v in jax.tree.leaves(state) if hasattr(v, "size"))


def flops_of(fn, *args) -> float:
    return jax.jit(fn).lower(*args).compile().cost_analysis().get("flops", float("nan"))


def gpt_stats(n_layer: int, n_head: int, d_model: int, token_ids) -> tuple[int, float]:
    cfg = GPTConfig(pos_method="rope", block_size=CONTEXT_LEN, vocab_size=VOCAB,
                     n_layer=n_layer, n_head=n_head, n_embd=d_model)
    m = GPT(cfg, rngs=nnx.Rngs(0))
    p = count_params(m)
    f = flops_of(lambda idx, t: cross_entropy(m(idx)[:, :-1, :], t), token_ids, token_ids[:, 1:])
    return p, f


def summ_stats(Ks: tuple, n_layers: int, n_heads: int, d_model: int, token_ids) -> tuple[int, float, int]:
    cfg = SummConfig(Ks=Ks, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                      pos_method="rope", context_len=CONTEXT_LEN, vocab_size=VOCAB)
    m = SummTransformer(cfg, rngs=nnx.Rngs(0))
    p = count_params(m)
    f = flops_of(lambda idx: m(idx)[0], token_ids)
    n_fuse = len(Ks)
    eff_depth = n_layers * (1 + 2 * n_fuse)
    return p, f, eff_depth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=sorted(TARGETS), default="medium",
                     help="gpt2 baseline to size against (sets d_model/n_heads, and the reference row)")
    ap.add_argument("--gpt-n-layer", type=int, default=None,
                     help="override the reference gpt2's depth (default: --target's own preset "
                          "depth, e.g. 24 for medium) -- d_model/n_heads still come from --target. "
                          "Use this to sanity-check against a hypothetical depth not tied to any "
                          "real config in configs/gpt2_jax/; irrelevant for matching an actual run.")
    ap.add_argument("--ks-lengths", type=str, default="2,3,4",
                     help="comma list of n_fuse values (len(Ks)) to try")
    ap.add_argument("--ks-values", type=str, default="2,4",
                     help="comma list of per-stage K values to try (uniform Ks=(k,)*n_fuse per combo)")
    ap.add_argument("--n-layers", type=str, default="1,2,3",
                     help="comma list of n_layers values to try")
    args = ap.parse_args()

    n_layer_base, n_head, d_model = TARGETS[args.target]
    if args.gpt_n_layer is not None:
        n_layer_base = args.gpt_n_layer
    ks_lengths = [int(x) for x in args.ks_lengths.split(",")]
    ks_values = [int(x) for x in args.ks_values.split(",")]
    n_layers_list = [int(x) for x in args.n_layers.split(",")]

    token_ids = jax.random.randint(jax.random.PRNGKey(0), (1, CONTEXT_LEN), 0, VOCAB)

    print(f"=== gpt2-{args.target} reference (n_layer={n_layer_base}, n_head={n_head}, d_model={d_model}) ===")
    gp, gf = gpt_stats(n_layer_base, n_head, d_model, token_ids)
    gpt_kv_mb = 2 * n_layer_base * CONTEXT_LEN * d_model * DTYPE_BYTES / 1e6
    print(f"params={gp:,}  flops={gf:,.0f}  kv_cache={gpt_kv_mb:.1f}MB")
    print()

    rows = []
    for n_fuse, k_val, n_layers in itertools.product(ks_lengths, ks_values, n_layers_list):
        Ks = (k_val,) * n_fuse
        if CONTEXT_LEN // (k_val ** n_fuse) < 1:
            continue  # Ks too aggressive for this context_len -- top level would be empty
        sp, sf, eff_depth = summ_stats(Ks, n_layers, n_head, d_model, token_ids)
        summ_kv_mb = 2 * (1 + n_fuse) * n_layers * CONTEXT_LEN * d_model * DTYPE_BYTES / 1e6
        rows.append(dict(
            Ks=Ks, n_layers=n_layers, n_fuse=n_fuse, eff_depth=eff_depth,
            params=sp, param_delta=100 * (sp - gp) / gp,
            flops=sf, flops_ratio=sf / gf,
            kv_mb=summ_kv_mb, kv_ratio=summ_kv_mb / gpt_kv_mb,
        ))

    rows.sort(key=lambda r: abs(r["param_delta"]))
    print(f"=== summformer-{args.target} sweep ({len(rows)} combos, sorted by closest param match) ===")
    header = f"{'Ks':<14}{'n_layers':>9}{'eff_depth':>10}{'params':>14}{'Δparams':>9}{'flops_ratio':>12}{'kv_ratio':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{str(r['Ks']):<14}{r['n_layers']:>9}{r['eff_depth']:>10}{r['params']:>14,}"
              f"{r['param_delta']:>+8.1f}%{r['flops_ratio']:>11.3f}x{r['kv_ratio']:>9.3f}x")


if __name__ == "__main__":
    main()
