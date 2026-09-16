"""CPU-only diagnostic: teacher-forced dense decode vs incremental KV-cache decode consistency
check on a real train sample. No checkpoint needed (random-init weights) -- this tests
implementation correctness of decode_generate's KV cache against decode_logits_and_target's dense
path, not trained accuracy. If KV-cache hidden states diverge from dense, also runs a third
"full recompute, no cache" (growing-prefix truncate+rerun) variant to isolate whether the bug is
in Attention.step/chunk_step's cache bookkeeping specifically, vs. the group/padding construction
shared by both incremental paths. Usage: JAX_PLATFORMS=cpu python3 -m image_lagcodec.scripts.kv_consistency_check
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from image_lagcodec.run_lagcodec import (
    Config, HierEncDec, load_config_module, CONFIG_FIELDS, load_cifar10,
    images_to_positions, pixel_order_for, code_embed_proj, run_block,
)
import image_lagcodec.eqx_common as eqx_common

if jax.default_backend() == "cpu":
    # splash_attention is a TPU-only Pallas kernel (no CPU interpret path wired up here) -- swap
    # in a plain jnp causal-softmax attention with identical math for the dense/batched path only.
    # The KV-cache .step/.chunk_step path already uses plain einsum, unaffected.
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

CONFIG_PATH = REPO_ROOT / "image_lagcodec" / "configs" / "overfit1k_baseline.py"
B = 4  # small real batch


def build_cfg():
    mod = load_config_module(CONFIG_PATH)
    kwargs = {k: mod[k] for k in CONFIG_FIELDS if k in mod}
    return Config(**kwargs)


def dense_h_t(level, target_seq, ctx_idx, decoder_ncodes):
    """Replicates decode_logits_and_target's internals but also returns h_t (pre-head hidden
    states) and the constructed xe (needed for the full-recompute control variant)."""
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    Bc = target_seq.shape[0]
    D = level.bos_embed.shape[-1]
    te = level._dec_embed_target(target_seq)
    n_blocks = ctx_idx.shape[1]
    G = decoder_ncodes
    pad_blocks = (-n_blocks) % G
    n_blocks_p = n_blocks + pad_blocks
    n_groups = n_blocks_p // G
    ctx_tok = code_embed_proj(ctx_idx, level.ctx_embed, level.ctx_proj)
    if pad_blocks > 0:
        ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
        te = jnp.pad(te, ((0, 0), (0, pad_blocks * level.K), (0, 0)))
    ctx_g = ctx_tok.reshape(Bc, n_groups, G, D)
    te_g = te.reshape(Bc, n_groups, G * level.K, D)
    bos_g = jnp.broadcast_to(level.bos_embed, (Bc, n_groups, 1, D))
    per_group_len = G + 1 + G * level.K
    xe = jnp.concatenate([ctx_g, bos_g, te_g], axis=2).reshape(Bc, n_groups * per_group_len, D)
    x = xe
    for blk in blocks:
        x = run_block(blk, x, level.remat)
    h_full = ln_f(x)
    pred_pos = (jnp.arange(n_groups)[:, None] * per_group_len + G
                + jnp.arange(G * level.K)[None, :]).reshape(-1)
    n_blocks_orig = ctx_idx.shape[1]
    valid_len = n_blocks_orig * level.K
    h_t = h_full[:, pred_pos, :][:, :valid_len, :]
    return h_t, xe, blocks, ln_f, pred_pos, valid_len, per_group_len, n_groups


def incremental_kv_h_t(level, target_seq, ctx_idx, decoder_ncodes):
    """Teacher-forced mirror of decode_generate's group_step loop -- feeds REAL target_seq values
    (not sampled ones) as x_input at every position, collecting h at each prediction point."""
    blocks, ln_f = level._dec_blocks(), level._dec_ln_f()
    Bc, n_blocks, _ = ctx_idx.shape
    D = level.bos_embed.shape[-1]
    hd = D // level.n_heads
    G = decoder_ncodes
    pad_blocks = (-n_blocks) % G
    n_blocks_p = n_blocks + pad_blocks
    n_groups = n_blocks_p // G
    per_group_len = G + 1 + G * level.K
    L_total = n_groups * per_group_len
    ctx_tok = code_embed_proj(ctx_idx, level.ctx_embed, level.ctx_proj)
    if pad_blocks > 0:
        ctx_tok = jnp.pad(ctx_tok, ((0, 0), (0, pad_blocks), (0, 0)))
    ctx_tok_g = ctx_tok.reshape(Bc, n_groups, G, D)

    valid_len = n_blocks * level.K
    tgt_pad = (-valid_len) % (G * level.K)
    target_p = target_seq
    if tgt_pad > 0:
        pad_shape = list(target_seq.shape)
        pad_shape[1] = tgt_pad
        target_p = jnp.concatenate([target_seq, jnp.zeros(pad_shape, dtype=target_seq.dtype)], axis=1)
    target_g = target_p.reshape(Bc, n_groups, G * level.K, *target_seq.shape[2:])

    def self_step(x_new, ck, cv, pos):
        new_ck, new_cv = [], []
        x = x_new
        for i, blk in enumerate(blocks):
            x, ck_i, cv_i = blk.step(x, ck[i], cv[i], pos, L_total)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

    def self_chunk_step(x_chunk, ck, cv, pos_start):
        new_ck, new_cv = [], []
        x = x_chunk
        for i, blk in enumerate(blocks):
            x, ck_i, cv_i = blk.chunk_step(x, ck[i], cv[i], pos_start, L_total)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return ln_f(x), jnp.stack(new_ck), jnp.stack(new_cv)

    cache_k = jnp.zeros((len(blocks), Bc, level.n_kv_heads, L_total, hd))
    cache_v = jnp.zeros_like(cache_k)
    pos = 0
    h_all = []
    for g in range(n_groups):
        bos_in = jnp.broadcast_to(level.bos_embed, (Bc, 1, D))
        chunk = jnp.concatenate([ctx_tok_g[:, g], bos_in], axis=1)
        h_chunk, cache_k, cache_v = self_chunk_step(chunk, cache_k, cache_v, pos)
        pos += (G + 1)
        h = h_chunk[:, -1, :]
        h_all.append(h)
        gt = target_g[:, g]  # (Bc, G*K, ...) real ground-truth members for this group
        x_input = level._dec_embed_target(gt[:, 0])
        for m in range(1, G * level.K):
            h, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
            pos += 1
            h_all.append(h)
            x_input = level._dec_embed_target(gt[:, m])
        _, cache_k, cache_v = self_step(x_input, cache_k, cache_v, pos)
        pos += 1
    h_t = jnp.stack(h_all, axis=1)  # (Bc, n_groups*G*K, D)
    return h_t[:, :valid_len, :]


def full_recompute_h_t(level, xe, blocks, ln_f, pred_pos, valid_len, remat):
    """Growing-prefix, NO cache: for each prediction position, truncate the (already-correctly-
    built) dense xe to [:pos+1] and rerun the block stack densely from scratch. Isolates whether
    a divergence is in the KV-cache step/chunk_step bookkeeping specifically (this path uses none
    of that) vs. something in the shared group/padding construction (would show up here too)."""
    h_out = []
    for pos in [int(p) for p in pred_pos[:valid_len]]:
        prefix = xe[:, :pos + 1, :]
        x = prefix
        for blk in blocks:
            x = run_block(blk, x, remat)
        h_out.append(ln_f(x)[:, -1, :])
    return jnp.stack(h_out, axis=1)


def main():
    cfg = build_cfg()
    key = jax.random.PRNGKey(0)
    model = HierEncDec(key, cfg)
    level = model.levels[0]

    (train_np, train_labels), _ = load_cifar10(Path(os.path.expanduser("~/qcute/datasets")))
    pixel_order = pixel_order_for(cfg)
    imgs = train_np[:B]
    flat_bytes = jnp.array(images_to_positions(imgs, cfg, pixel_order))

    x_in = code_embed_proj(flat_bytes, level.own_input_embed, level.own_input_proj)
    enc_out = level.encode(x_in, flat_bytes, rng=None)  # rng=None -> deterministic quantize_hard
    ctx_idx = enc_out["code_idx"]
    target_seq = flat_bytes
    G = cfg.decoder_ncodes[0]

    h_dense, xe, blocks, ln_f, pred_pos, valid_len, per_group_len, n_groups = dense_h_t(
        level, target_seq, ctx_idx, G)
    h_kv = incremental_kv_h_t(level, target_seq, ctx_idx, G)

    diff = jnp.abs(h_dense - h_kv)
    per_pos_max = diff.max(axis=(0, 2))  # (valid_len,)
    tol = 1e-3
    divergent = jnp.where(per_pos_max > tol)[0]
    print(f"n_groups={n_groups} G={G} K={level.K} valid_len={valid_len}")
    print(f"dense-vs-kv-cache: max_abs_diff={float(diff.max()):.6f} mean_abs_diff={float(diff.mean()):.6f}")
    if divergent.size == 0:
        print(f"dense-vs-kv-cache: CONSISTENT (all positions within tol={tol})")
    else:
        first = int(divergent[0])
        group_idx, in_group = divmod(first, G * level.K)
        print(f"dense-vs-kv-cache: DIVERGES starting at flat position {first} "
              f"(group {group_idx}, offset {in_group} within group), "
              f"{divergent.size}/{valid_len} positions exceed tol={tol}")
        print("running full-recompute (no-cache, growing-prefix) control variant...")
        h_recompute = full_recompute_h_t(level, xe, blocks, ln_f, pred_pos, valid_len, level.remat)
        diff2 = jnp.abs(h_dense - h_recompute)
        per_pos_max2 = diff2.max(axis=(0, 2))
        divergent2 = jnp.where(per_pos_max2 > tol)[0]
        print(f"dense-vs-full-recompute: max_abs_diff={float(diff2.max()):.6f} "
              f"mean_abs_diff={float(diff2.mean()):.6f}")
        if divergent2.size == 0:
            print("dense-vs-full-recompute: CONSISTENT -> bug isolated to KV-cache step/chunk_step "
                  "bookkeeping (Attention.step/chunk_step pos tracking, dynamic_update_slice, or "
                  "RoPE position offset), NOT the group/padding construction shared by both paths.")
        else:
            first2 = int(divergent2[0])
            g2, o2 = divmod(first2, G * level.K)
            print(f"dense-vs-full-recompute: ALSO diverges starting at flat position {first2} "
                  f"(group {g2}, offset {o2}) -> bug is in the shared group/padding/position "
                  "construction (or causal-masking assumption), not specific to the KV-cache math.")


if __name__ == "__main__":
    main()
