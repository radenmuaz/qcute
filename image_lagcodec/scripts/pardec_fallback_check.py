"""CPU-only check: decoder_ncodes>=n_blocks fallback triggers correctly and matches the original
decode exactly (it literally calls it); Config warnings fire without crashing construction."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
import warnings
from pathlib import Path
import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from image_lagcodec.run_lagcodec import Config, HierEncDec, code_embed_proj
import image_lagcodec.eqx_common as eqx_common

if jax.default_backend() == "cpu":
    def _cpu_dense_attention(q, k, v, causal, sm_scale):
        Bc, Hq, T, hd = q.shape
        Hkv = k.shape[1]
        n_rep = Hq // Hkv
        if n_rep > 1:
            k = jnp.repeat(k, n_rep, axis=1)
            v = jnp.repeat(v, n_rep, axis=1)
        logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * sm_scale
        if causal:
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            logits = jnp.where(mask[None, None], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bhts,bhsd->bhtd", attn, v)
    eqx_common.splash_attention = _cpu_dense_attention

# 1. Config warnings should fire, not crash, for a deliberately wasteful/redundant setup.
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cfg_bad = Config(
        d_model=(64, 64, 64, 64), n_layers=(1, 1, 1, 1), n_heads=(2, 2, 2, 2),
        strides=(4, 4, 4, -1), code_vocab=(16, 16, 16, 16), pq_chunks=(4, 4, 4, 4),
        pq_dim=(32, 16, 16, 16), byte_group=3, token_head_type="linears", mtp_horizon=1,
        decoder_ncodes=(1, 4, 4, 4), ncodes_window=(-1, 0, 0, 0),   # level0: G=1, N=-1 -> should warn
        weight_sharing=True, curriculum_mode="no_freeze",
    )
    msgs = [str(x.message) for x in w]
    print(f"warnings fired: {len(msgs)}")
    for m in msgs:
        print(" -", m[:100])
    assert any("naive unbounded" in m for m in msgs), "expected the small-G+unbounded warning"
print("PASS: warnings fire without crashing construction\n")

# 2. G>=n_blocks fallback: build a config where level0's decoder_ncodes == n_blocks (256), confirm
#    decode_generate_pardec's output is IDENTICAL to decode_generate's (same call, bit-exact).
G = 256   # == n_blocks for level0 at this cfg (n_positions=1024, K=4 -> n_blocks=256)
cfg = Config(
    d_model=(64, 64, 64, 64), n_layers=(2, 2, 2, 2), n_heads=(2, 2, 2, 2),
    strides=(4, 4, 4, -1), code_vocab=(16, 16, 16, 16), pq_chunks=(4, 4, 4, 4),
    pq_dim=(32, 16, 16, 16), byte_group=3, token_head_type="linears", mtp_horizon=1,
    decoder_ncodes=(G, 4, 4, 4), ncodes_window=(2, 0, 0, 0),   # N irrelevant here, should be ignored
    weight_sharing=True, curriculum_mode="no_freeze",
)
key = jax.random.PRNGKey(0)
model = HierEncDec(key, cfg)
level = model.levels[0]

B, n_blocks = 2, 256
ctx_idx = jax.random.randint(jax.random.PRNGKey(1), (B, n_blocks, cfg.pq_chunks[0]), 0, cfg.code_vocab[0])
out_pardec = level.decode_generate_pardec(ctx_idx, G, greedy=True, seed=0)
out_orig = level.decode_generate(ctx_idx, G, greedy=True, seed=0)
print(f"decode_generate_pardec shape={out_pardec.shape} decode_generate shape={out_orig.shape}")
assert out_pardec.shape == out_orig.shape
assert bool(jnp.all(out_pardec == out_orig)), "fallback output should be BIT-EXACT (same call)"
print("PASS: G>=n_blocks fallback is bit-exact vs decode_generate")
