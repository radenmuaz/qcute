"""Equinox port of run_causalattn_v1.py (plain-dict-pytree JAX). Same architecture: 3-level
CAUSAL hierarchy over the flattened image with PQ-quantized categorical codes (gumbel-hard
STE), NTP heads on levels 1/2 so they double as free-running KV-cached generative priors,
and the same causal lag-1-row column-batched GQA decoder as the no-quant variant, except
looking codes up through embedding tables instead of a continuous vector. Only the parameter
representation changed to eqx.Module classes; data loading, CLI/config plumbing, and
save_sample_grid/quantize_hard/codebook_utilization/code_embed are imported unchanged from
run_causalattn_v1. Adds checkpoint save/resume (v1 had none). ColMix is duplicated here
(not shared via eqx_common) matching this codebase's single-file hard-fork convention --
each run_*.py stays self-contained.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from image_gen_cifar_jax.eqx_common import Attention, Block, RMSNorm, load_checkpoint, save_checkpoint
from image_gen_cifar_jax.run_causalattn_v1 import (
    SLOT_B, SLOT_G, SLOT_L0, SLOT_L0_MTP, SLOT_L1, SLOT_L2, SLOT_R, SLOT_RGB_MTP,
    BatchIterator, Config, Logger, MODULE_DIR, REPO_ROOT, code_embed, codebook_utilization,
    load_cifar10, load_config_module, quantize_hard, save_sample_grid, warmup_schedule,
    write_resolved_config,
)


def reshape_pq(logits: jnp.ndarray, pq_chunks: int, code_vocab: int) -> jnp.ndarray:
    return logits.reshape(*logits.shape[:-1], pq_chunks, code_vocab)


class ColMix(eqx.Module):
    norm: RMSNorm
    qkv: jnp.ndarray
    out: jnp.ndarray
    n_heads: int = eqx.field(static=True)
    group_size: int = eqx.field(static=True)

    def __init__(self, key, d_model: int, n_heads: int, group_size: int):
        k1, k2 = jax.random.split(key)
        self.norm = RMSNorm(d_model)
        self.qkv = jax.random.normal(k1, (d_model, 3 * d_model)) * 0.02
        self.out = jax.random.normal(k2, (d_model, d_model)) * 0.02
        self.n_heads, self.group_size = n_heads, group_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.group_size <= 1:
            return x
        B, R, C, D = x.shape
        g = self.group_size
        hd = D // self.n_heads
        xn = self.norm(x)
        qkv = xn @ self.qkv
        qkv = qkv.reshape(B, R, C // g, g, 3, self.n_heads, hd)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]

        def fold(t):
            t = jnp.moveaxis(t, -2, 3)
            return t.reshape(B * R * (C // g), self.n_heads, g, hd)

        qf, kf, vf = fold(q), fold(k), fold(v)
        scale = 1.0 / jnp.sqrt(hd).astype(jnp.float32)
        logits = jnp.einsum("bhqd,bhkd->bhqk", qf, kf) * scale
        attn = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum("bhqk,bhkd->bhqd", attn, vf)
        y = y.reshape(B, R, C // g, self.n_heads, g, hd)
        y = jnp.moveaxis(y, 3, -2).reshape(B, R, C, D)
        return x + y @ self.out


# ---------------------------------------------------------------------------
# Encoder: 3-level CAUSAL hierarchy, PQ-quantized, with NTP heads on levels 1/2
# ---------------------------------------------------------------------------

class EncoderLevel(eqx.Module):
    blocks: list
    ln_f: RMSNorm
    code_head: jnp.ndarray
    ntp_head: jnp.ndarray | None
    pq_chunks: int = eqx.field(static=True)
    code_vocab: int = eqx.field(static=True)
    code_extract_mode: str = eqx.field(static=True)

    def __init__(self, key, cfg: Config, has_ntp: bool):
        D = cfg.d_model
        k_blocks, k_head, k_ntp = jax.random.split(key, 3)
        block_keys = jax.random.split(k_blocks, cfg.n_layers)
        self.blocks = [Block(k, D, cfg.n_heads, cfg.n_kv_heads, cfg.mlp_mult, cfg.rope_base) for k in block_keys]
        self.ln_f = RMSNorm(D)
        self.code_head = jax.random.normal(k_head, (D, cfg.pq_chunks * cfg.code_vocab)) * 0.02
        self.ntp_head = jax.random.normal(k_ntp, (D, cfg.pq_chunks * cfg.code_vocab)) * 0.02 if has_ntp else None
        self.pq_chunks, self.code_vocab, self.code_extract_mode = cfg.pq_chunks, cfg.code_vocab, cfg.code_extract_mode

    def run(self, x: jnp.ndarray) -> jnp.ndarray:
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def pool(self, h: jnp.ndarray, stride: int) -> jnp.ndarray:
        M, L, D = h.shape
        h = h.reshape(M, L // stride, stride, D)
        return jnp.mean(h, axis=2) if self.code_extract_mode == "mean" else h[:, :, -1, :]

    def encode(self, x: jnp.ndarray, stride: int) -> tuple:
        h = self.run(x)
        pooled = self.pool(h, stride)
        logits = reshape_pq(pooled @ self.code_head, self.pq_chunks, self.code_vocab)
        return quantize_hard(logits)

    def ntp_logits(self, h: jnp.ndarray) -> jnp.ndarray:
        return reshape_pq(h @ self.ntp_head, self.pq_chunks, self.code_vocab)

    def ntp_loss(self, x: jnp.ndarray, target_idx: jnp.ndarray, cond: jnp.ndarray = None,
                 y_embed: jnp.ndarray = None) -> jnp.ndarray:
        if cond is not None:
            x = x + cond
        if y_embed is not None:
            x = x + y_embed[:, None, :]
        h = self.run(x)
        logits = self.ntp_logits(h[:, :-1, :])
        logp = jax.nn.log_softmax(logits, axis=-1)
        target = target_idx[:, 1:]
        return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))

    def step(self, x_new, cache_k, cache_v, pos, T_max) -> tuple:
        new_ck, new_cv = [], []
        x = x_new
        for i, blk in enumerate(self.blocks):
            x, ck_i, cv_i = blk.step(x, cache_k[i], cache_v[i], pos, T_max)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return x, jnp.stack(new_ck), jnp.stack(new_cv)


class ImageEncoder(eqx.Module):
    r_embed: jnp.ndarray
    g_embed: jnp.ndarray
    b_embed: jnp.ndarray
    level0: EncoderLevel
    code0_embed: jnp.ndarray
    level1: EncoderLevel
    code1_embed: jnp.ndarray
    level2: EncoderLevel
    level1_bos: jnp.ndarray
    level2_bos: jnp.ndarray
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        D = cfg.d_model
        keys = jax.random.split(key, 8)
        self.r_embed = jax.random.normal(keys[0], (256, D)) * 0.02
        self.g_embed = jax.random.normal(keys[1], (256, D)) * 0.02
        self.b_embed = jax.random.normal(keys[2], (256, D)) * 0.02
        self.level0 = EncoderLevel(keys[3], cfg, has_ntp=False)
        self.code0_embed = jax.random.normal(keys[4], (cfg.code_vocab, D)) * 0.02
        self.level1 = EncoderLevel(keys[5], cfg, has_ntp=True)
        self.code1_embed = jax.random.normal(keys[6], (cfg.code_vocab, D)) * 0.02
        self.level2 = EncoderLevel(keys[7], cfg, has_ntp=True)
        self.level1_bos = jnp.zeros((D,))
        self.level2_bos = jnp.zeros((D,))

    def __call__(self, r: jnp.ndarray, g: jnp.ndarray, b: jnp.ndarray, y_embed: jnp.ndarray = None) -> dict:
        cfg = self.cfg
        B = r.shape[0]
        r, g, b = r.reshape(B, -1), g.reshape(B, -1), b.reshape(B, -1)
        x0 = self.r_embed[r] + self.g_embed[g] + self.b_embed[b]
        code0_soft, code0_idx = self.level0.encode(x0, cfg.strides[0])

        x1 = code_embed(code0_soft, self.code0_embed)
        code1_soft, code1_idx = self.level1.encode(x1, cfg.strides[1])
        cond1 = jnp.repeat(code_embed(code1_idx, self.code1_embed), cfg.strides[1], axis=1)
        ntp1 = self.level1.ntp_loss(x1, code0_idx, cond=cond1, y_embed=y_embed)

        x2 = code_embed(code1_soft, self.code1_embed)
        code2_soft, code2_idx = self.level2.encode(x2, cfg.strides[2])
        ntp2 = self.level2.ntp_loss(x2, code1_idx, y_embed=y_embed)

        vocab = cfg.code_vocab
        return dict(code0_soft=code0_soft, code1_soft=code1_soft, code2_soft=code2_soft, ntp_loss=ntp1 + ntp2,
                    util0=codebook_utilization(code0_idx, vocab), util1=codebook_utilization(code1_idx, vocab),
                    util2=codebook_utilization(code2_idx, vocab))


# ---------------------------------------------------------------------------
# Decoder: causal, lag-1-row, GQA, column-batched, BOS-bootstrapped, code-embedding tables
# ---------------------------------------------------------------------------

class Decoder(eqx.Module):
    dec_l2_embed: jnp.ndarray
    dec_l1_embed: jnp.ndarray
    dec_l0_embed: jnp.ndarray
    byte_embed: jnp.ndarray
    slot_embed: jnp.ndarray
    bos_l2: jnp.ndarray
    bos_l1: jnp.ndarray
    bos_l0: jnp.ndarray
    col_mix: ColMix
    blocks: list
    ln_f: RMSNorm
    head_r: jnp.ndarray
    head_g: jnp.ndarray
    head_b: jnp.ndarray
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        D = cfg.d_model
        n_slots = 4 if cfg.decoder_mode == "mtp" else 6
        keys = jax.random.split(key, 10)
        self.dec_l2_embed = jax.random.normal(keys[0], (cfg.code_vocab, D)) * 0.02
        self.dec_l1_embed = jax.random.normal(keys[1], (cfg.code_vocab, D)) * 0.02
        self.dec_l0_embed = jax.random.normal(keys[2], (cfg.code_vocab, D)) * 0.02
        self.byte_embed = jax.random.normal(keys[3], (256, D)) * 0.02
        self.slot_embed = jax.random.normal(keys[4], (n_slots, D)) * 0.02
        self.bos_l2 = jnp.zeros((D,))
        self.bos_l1 = jnp.zeros((D,))
        self.bos_l0 = jnp.zeros((D,))
        self.col_mix = ColMix(keys[5], D, cfg.n_heads, cfg.col_group_size)
        block_keys = jax.random.split(keys[6], cfg.n_layers)
        self.blocks = [Block(k, D, cfg.n_heads, cfg.n_kv_heads, cfg.mlp_mult, cfg.rope_base) for k in block_keys]
        self.ln_f = RMSNorm(D)
        self.head_r = jax.random.normal(keys[7], (D, 256)) * 0.02
        self.head_g = jax.random.normal(keys[8], (D, 256)) * 0.02
        self.head_b = jax.random.normal(keys[9], (D, 256)) * 0.02

    def lagged_code_embeds(self, code2, code1, code0, y_embed=None) -> tuple:
        cfg = self.cfg
        img, D = cfg.img_size, cfg.d_model
        l2e = code_embed(code2, self.dec_l2_embed)
        l1e = code_embed(code1, self.dec_l1_embed).reshape(-1, img, code1.shape[1] // img, D)
        l0e = code_embed(code0, self.dec_l0_embed).reshape(-1, img, code0.shape[1] // img, D)
        B = l2e.shape[0]
        bos_l2 = jnp.broadcast_to(self.bos_l2, (B, 1, D))
        bos_l1 = jnp.broadcast_to(self.bos_l1, (B, 1, l1e.shape[2], D))
        bos_l0 = jnp.broadcast_to(self.bos_l0, (B, 1, l0e.shape[2], D))
        l2e_lag = jnp.concatenate([bos_l2, l2e[:, :-1]], axis=1)
        l1e_lag = jnp.concatenate([bos_l1, l1e[:, :-1]], axis=1)
        l0e_lag = jnp.concatenate([bos_l0, l0e[:, :-1]], axis=1)
        if y_embed is not None:
            l2e_lag = l2e_lag + y_embed[:, None, :]
            l1e_lag = l1e_lag + y_embed[:, None, None, :]
            l0e_lag = l0e_lag + y_embed[:, None, None, :]
        return l2e_lag, l1e_lag, l0e_lag

    def per_column_cond(self, code2, code1, code0, y_embed=None) -> tuple:
        img = self.cfg.img_size
        l2e_lag, l1e_lag, l0e_lag = self.lagged_code_embeds(code2, code1, code0, y_embed)
        B, _, D = l2e_lag.shape
        n_l1, n_l0 = l1e_lag.shape[2], l0e_lag.shape[2]
        cols = jnp.arange(img)
        l1_g, l0_g = cols // (img // n_l1), cols // (img // n_l0)
        l2e_col = jnp.broadcast_to(l2e_lag[:, :, None, :], (B, img, img, D))
        l1e_col = l1e_lag[:, :, l1_g, :]
        l0e_col = l0e_lag[:, :, l0_g, :]
        return self.col_mix(l2e_col), self.col_mix(l1e_col), self.col_mix(l0e_col)

    def __call__(self, code2, code1, code0, r, g, b, y_embed=None) -> tuple:
        cfg = self.cfg
        img, D = cfg.img_size, cfg.d_model
        B = r.shape[0]
        l2e_col, l1e_col, l0e_col = self.per_column_cond(code2, code1, code0, y_embed)
        r_e, g_e, b_e = self.byte_embed[r], self.byte_embed[g], self.byte_embed[b]

        if cfg.decoder_mode == "mtp":
            slots = jnp.stack([l2e_col, l1e_col, l0e_col, r_e + g_e + b_e], axis=3)
        else:
            slots = jnp.stack([l2e_col, l1e_col, l0e_col, r_e, g_e, b_e], axis=3)
        n_slots = slots.shape[3]
        slots = slots + self.slot_embed[None, None, None, :, :]

        x = jnp.transpose(slots, (0, 2, 1, 3, 4)).reshape(B * img, img * n_slots, D)
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_f(x)
        h = h.reshape(B, img, img, n_slots, D)
        h = jnp.transpose(h, (0, 2, 1, 3, 4))

        if cfg.decoder_mode == "mtp":
            h_seed = h[:, :, :, SLOT_L0_MTP, :]
            logits_r, logits_g, logits_b = h_seed @ self.head_r, h_seed @ self.head_g, h_seed @ self.head_b
        else:
            logits_r = h[:, :, :, SLOT_L0, :] @ self.head_r
            logits_g = h[:, :, :, SLOT_R, :] @ self.head_g
            logits_b = h[:, :, :, SLOT_G, :] @ self.head_b

        def ce(logits, target):
            logp = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))

        loss_r, loss_g, loss_b = ce(logits_r, r), ce(logits_g, g), ce(logits_b, b)
        acc = (jnp.mean(jnp.argmax(logits_r, -1) == r) + jnp.mean(jnp.argmax(logits_g, -1) == g)
               + jnp.mean(jnp.argmax(logits_b, -1) == b)) / 3
        return (loss_r + loss_g + loss_b) / 3, acc

    def step(self, x_new, cache_k, cache_v, pos, T_max) -> tuple:
        new_ck, new_cv = [], []
        x = x_new
        for i, blk in enumerate(self.blocks):
            x, ck_i, cv_i = blk.step(x, cache_k[i], cache_v[i], pos, T_max)
            new_ck.append(ck_i)
            new_cv.append(cv_i)
        return x, jnp.stack(new_ck), jnp.stack(new_cv)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CausalAttnModel(eqx.Module):
    encoder: ImageEncoder
    decoder: Decoder
    class_embed: jnp.ndarray | None
    cfg: Config = eqx.field(static=True)

    def __init__(self, key, cfg: Config):
        self.cfg = cfg
        k1, k2, k3 = jax.random.split(key, 3)
        self.encoder = ImageEncoder(k1, cfg)
        self.decoder = Decoder(k2, cfg)
        self.class_embed = jax.random.normal(k3, (cfg.n_classes, cfg.d_model)) * 0.02 if cfg.class_conditional else None

    def __call__(self, r: jnp.ndarray, g: jnp.ndarray, b: jnp.ndarray, y: jnp.ndarray) -> tuple:
        cfg = self.cfg
        y_embed = self.class_embed[y] if cfg.class_conditional else None
        enc = self.encoder(r, g, b, y_embed)
        loss, acc = self.decoder(enc["code2_soft"], enc["code1_soft"], enc["code0_soft"], r, g, b, y_embed)
        total = loss + cfg.ntp_aux_weight * enc["ntp_loss"]
        bpb = loss / jnp.log(2.0)
        return total, (bpb, acc, enc["util0"], enc["util1"], enc["util2"])

    def sample_codes(self, B: int, greedy: bool = False, temperature: float = 1.0, y_embed: jnp.ndarray = None,
                      prompt_code1: jnp.ndarray = None, prompt_code0: jnp.ndarray = None, seed: int = 0) -> tuple:
        cfg = self.cfg
        enc = self.encoder
        code0_len = cfg.img_size * cfg.img_size // cfg.strides[0]
        code1_len = code0_len // cfg.strides[1]
        D = cfg.d_model
        hd = D // cfg.n_heads
        pqc = cfg.pq_chunks
        n_prompt1 = prompt_code1.shape[1] if prompt_code1 is not None else 0
        n_prompt0 = prompt_code0.shape[1] if prompt_code0 is not None else 0
        rng = jax.random.PRNGKey(seed)

        def sample(logits, key):
            if greedy:
                return jnp.argmax(logits, axis=-1)
            return jax.random.categorical(key, logits / temperature, axis=-1)

        Hkv = cfg.n_kv_heads

        cache_k = jnp.zeros((cfg.n_layers, B, Hkv, code1_len, hd))
        cache_v = jnp.zeros_like(cache_k)
        step2 = jax.jit(lambda x, ck, cv, pos: enc.level2.step(x, ck, cv, pos, code1_len))
        x_new = jnp.broadcast_to(enc.level2_bos, (B, D))
        if y_embed is not None:
            x_new = x_new + y_embed
        code1_idx = []
        for t in range(code1_len):
            h, cache_k, cache_v = step2(x_new, cache_k, cache_v, t)
            if t < n_prompt1:
                nxt = prompt_code1[:, t, :]
            else:
                h_n = enc.level2.ln_f(h)
                logits = reshape_pq(h_n @ enc.level2.ntp_head, pqc, cfg.code_vocab)
                rng, k = jax.random.split(rng)
                nxt = sample(logits, k)
            code1_idx.append(nxt)
            x_new = code_embed(nxt, enc.code1_embed)
            if y_embed is not None:
                x_new = x_new + y_embed
        code1_idx = jnp.stack(code1_idx, axis=1)

        cache_k = jnp.zeros((cfg.n_layers, B, Hkv, code0_len, hd))
        cache_v = jnp.zeros_like(cache_k)
        step1 = jax.jit(lambda x, ck, cv, pos: enc.level1.step(x, ck, cv, pos, code0_len))
        x_new = jnp.broadcast_to(enc.level1_bos, (B, D))
        if y_embed is not None:
            x_new = x_new + y_embed
        code0_idx = []
        for t in range(code0_len):
            h, cache_k, cache_v = step1(x_new, cache_k, cache_v, t)
            if t < n_prompt0:
                nxt = prompt_code0[:, t, :]
            else:
                h_n = enc.level1.ln_f(h)
                logits = reshape_pq(h_n @ enc.level1.ntp_head, pqc, cfg.code_vocab)
                rng, k = jax.random.split(rng)
                nxt = sample(logits, k)
            code0_idx.append(nxt)
            cond = code_embed(code1_idx[:, t // cfg.strides[1]], enc.code1_embed)
            x_new = code_embed(nxt, enc.code0_embed) + cond
            if y_embed is not None:
                x_new = x_new + y_embed
        code0_idx = jnp.stack(code0_idx, axis=1)

        x2 = code_embed(code1_idx, enc.code1_embed)
        _, code2_idx = enc.level2.encode(x2, cfg.strides[2])
        return code0_idx, code1_idx, code2_idx

    def encode_prompt_codes(self, prompt_r, prompt_g, prompt_b, n_prompt_rows: int, y_embed=None) -> tuple:
        img = self.cfg.img_size
        enc = self.encoder(prompt_r, prompt_g, prompt_b, y_embed)
        code1_idx_full = jnp.argmax(enc["code1_soft"], axis=-1)
        code0_idx_full = jnp.argmax(enc["code0_soft"], axis=-1)
        n1_per_row = code1_idx_full.shape[1] // img
        n0_per_row = code0_idx_full.shape[1] // img
        return code1_idx_full[:, :n_prompt_rows * n1_per_row, :], code0_idx_full[:, :n_prompt_rows * n0_per_row, :]

    def decoder_generate(self, code0_idx, code1_idx, code2_idx, n: int, greedy: bool = False,
                          temperature: float = 1.0, y_embed: jnp.ndarray = None, prompt_r=None, prompt_g=None,
                          prompt_b=None, seed: int = 0) -> jnp.ndarray:
        cfg = self.cfg
        dec = self.decoder
        img, D = cfg.img_size, cfg.d_model
        n_slots = 4 if cfg.decoder_mode == "mtp" else 6
        T_max = img * n_slots
        Bc = n * img
        hd = D // cfg.n_heads
        n_prompt = prompt_r.shape[1] if prompt_r is not None else 0
        rng = jax.random.PRNGKey(seed)

        l2e_col, l1e_col, l0e_col = dec.per_column_cond(code2_idx, code1_idx, code0_idx, y_embed)
        l2e = jnp.transpose(l2e_col, (0, 2, 1, 3)).reshape(Bc, img, D)
        l1e = jnp.transpose(l1e_col, (0, 2, 1, 3)).reshape(Bc, img, D)
        l0e = jnp.transpose(l0e_col, (0, 2, 1, 3)).reshape(Bc, img, D)

        cache_k = jnp.zeros((cfg.n_layers, Bc, cfg.n_kv_heads, T_max, hd))
        cache_v = jnp.zeros_like(cache_k)
        slot_w = dec.slot_embed
        step_fn = jax.jit(lambda x, ck, cv, pos: dec.step(x, ck, cv, pos, T_max))

        def sample(logits, key):
            if greedy:
                return jnp.argmax(logits, axis=-1)
            return jax.random.categorical(key, logits / temperature, axis=-1)

        r_out = jnp.zeros((n, img, img), dtype=jnp.int32)
        g_out = jnp.zeros((n, img, img), dtype=jnp.int32)
        b_out = jnp.zeros((n, img, img), dtype=jnp.int32)
        pos = 0

        for row in range(img):
            x, cache_k, cache_v = step_fn(l2e[:, row] + slot_w[SLOT_L2], cache_k, cache_v, pos); pos += 1
            x, cache_k, cache_v = step_fn(l1e[:, row] + slot_w[SLOT_L1], cache_k, cache_v, pos); pos += 1
            l0_slot = slot_w[SLOT_L0_MTP if cfg.decoder_mode == "mtp" else SLOT_L0]
            x, cache_k, cache_v = step_fn(l0e[:, row] + l0_slot, cache_k, cache_v, pos); pos += 1
            h_seed = dec.ln_f(x)

            if row < n_prompt:
                r_row = prompt_r[:, row, :].reshape(-1)
                g_row = prompt_g[:, row, :].reshape(-1)
                b_row = prompt_b[:, row, :].reshape(-1)
                if cfg.decoder_mode == "mtp":
                    rgb_e = dec.byte_embed[r_row] + dec.byte_embed[g_row] + dec.byte_embed[b_row] + slot_w[SLOT_RGB_MTP]
                    _, cache_k, cache_v = step_fn(rgb_e, cache_k, cache_v, pos); pos += 1
                else:
                    _, cache_k, cache_v = step_fn(dec.byte_embed[r_row] + slot_w[SLOT_R], cache_k, cache_v, pos); pos += 1
                    _, cache_k, cache_v = step_fn(dec.byte_embed[g_row] + slot_w[SLOT_G], cache_k, cache_v, pos); pos += 1
                    _, cache_k, cache_v = step_fn(dec.byte_embed[b_row] + slot_w[SLOT_B], cache_k, cache_v, pos); pos += 1
            elif cfg.decoder_mode == "mtp":
                rng, kr, kg, kb = jax.random.split(rng, 4)
                r_row = sample(h_seed @ dec.head_r, kr)
                g_row = sample(h_seed @ dec.head_g, kg)
                b_row = sample(h_seed @ dec.head_b, kb)
                rgb_e = dec.byte_embed[r_row] + dec.byte_embed[g_row] + dec.byte_embed[b_row] + slot_w[SLOT_RGB_MTP]
                _, cache_k, cache_v = step_fn(rgb_e, cache_k, cache_v, pos); pos += 1
            else:
                rng, kr, kg, kb = jax.random.split(rng, 4)
                r_row = sample(h_seed @ dec.head_r, kr)
                x, cache_k, cache_v = step_fn(dec.byte_embed[r_row] + slot_w[SLOT_R], cache_k, cache_v, pos); pos += 1
                h_r = dec.ln_f(x)
                g_row = sample(h_r @ dec.head_g, kg)
                x, cache_k, cache_v = step_fn(dec.byte_embed[g_row] + slot_w[SLOT_G], cache_k, cache_v, pos); pos += 1
                h_g = dec.ln_f(x)
                b_row = sample(h_g @ dec.head_b, kb)
                _, cache_k, cache_v = step_fn(dec.byte_embed[b_row] + slot_w[SLOT_B], cache_k, cache_v, pos); pos += 1

            r_out = r_out.at[:, row, :].set(r_row.reshape(n, img))
            g_out = g_out.at[:, row, :].set(g_row.reshape(n, img))
            b_out = b_out.at[:, row, :].set(b_row.reshape(n, img))

        return jnp.stack([r_out, g_out, b_out], axis=-1).clip(0, 255).astype(jnp.uint8)

    def generate(self, n: int, greedy: bool = False, temperature: float = 1.0, y: jnp.ndarray = None,
                 full_prompt_r=None, full_prompt_g=None, full_prompt_b=None, n_prompt: int = 0,
                 seed: int = 0) -> jnp.ndarray:
        cfg = self.cfg
        y_embed = self.class_embed[y] if (cfg.class_conditional and y is not None) else None
        prompt_code1 = prompt_code0 = None
        prompt_r = prompt_g = prompt_b = None
        if n_prompt > 0:
            prompt_code1, prompt_code0 = self.encode_prompt_codes(full_prompt_r, full_prompt_g, full_prompt_b,
                                                                    n_prompt, y_embed)
            prompt_r = full_prompt_r[:, :n_prompt, :]
            prompt_g = full_prompt_g[:, :n_prompt, :]
            prompt_b = full_prompt_b[:, :n_prompt, :]
        code0_idx, code1_idx, code2_idx = self.sample_codes(n, greedy=greedy, temperature=temperature,
                                                              y_embed=y_embed, prompt_code1=prompt_code1,
                                                              prompt_code0=prompt_code0, seed=seed)
        return self.decoder_generate(code0_idx, code1_idx, code2_idx, n, greedy=greedy, temperature=temperature,
                                      y_embed=y_embed, prompt_r=prompt_r, prompt_g=prompt_g, prompt_b=prompt_b,
                                      seed=seed)


def save_compare_grid(gen: np.ndarray, gt: np.ndarray, path: Path, pad: int = 2) -> None:
    """gen/gt: (n,H,W,3) uint8 -- side-by-side [generated | ground truth] pairs, one row per
    example, for visually confirming overfitting (generated should closely match gt when the
    model has memorized a small training subset)."""
    from PIL import Image
    n, h, w, c = gen.shape
    grid = np.full((n * (h + pad) + pad, 2 * (w + pad) + pad, c), 255, dtype=np.uint8)
    for i in range(n):
        y = pad + i * (h + pad)
        grid[y:y + h, pad:pad + w] = gen[i]
        grid[y:y + h, 2 * pad + w:2 * pad + 2 * w] = gt[i]
    Image.fromarray(grid).save(path)


def count_params(model: CausalAttnModel) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return sum(x.size for x in leaves)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_train_step(optimizer):
    def loss_fn(model, r, g, b, y):
        return model(r, g, b, y)

    def train_step(model, opt_state, r, g, b, y):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, r, g, b, y)
        grads = jax.lax.pmean(grads, axis_name="d")
        aux = jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, aux

    return jax.pmap(train_step, axis_name="d")


def make_eval_step():
    def eval_step(model, r, g, b, y):
        _, aux = model(r, g, b, y)
        return jax.tree_util.tree_map(lambda a: jax.lax.pmean(a, axis_name="d"), aux)

    return jax.pmap(eval_step, axis_name="d")


CONFIG_FIELDS = ("d_model", "n_layers", "n_heads", "code_vocab", "pq_chunks", "decoder_mode",
                  "col_group_size", "class_conditional", "n_classes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True,
                    help="Python config file (image_gen_cifar_jax/configs/*.py) -- every run must have one")
    p.add_argument("--data_root", type=str, default=str(REPO_ROOT / "datasets"))
    p.add_argument("--run_name", type=str, default="cifar_causalattn_jax")
    p.add_argument("--batch_size", type=int, default=8, help="per-device batch size")
    p.add_argument("--n_devices", type=int, default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every_epochs", type=int, default=1)
    p.add_argument("--checkpoint_every_epochs", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--train_subset_n", type=int, default=None,
                    help="truncate the train split to the first N images -- for overfit sanity checks")
    p.add_argument("--qual_gen_n", type=int, default=4)
    p.add_argument("--qual_gen_greedy", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--qual_gen_temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--code_vocab", type=int, default=16)
    p.add_argument("--pq_chunks", type=int, default=4)
    p.add_argument("--decoder_mode", type=str, default="seq", choices=["seq", "mtp"])
    p.add_argument("--col_group_size", type=int, default=1)
    p.add_argument("--class_conditional", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--n_classes", type=int, default=10)

    pre_args, _ = p.parse_known_args()
    config_vars = load_config_module(pre_args.config)
    known = {a.dest for a in p._actions}
    unknown = set(config_vars) - known
    if unknown:
        p.error(f"--config {pre_args.config} sets unknown field(s): {sorted(unknown)}")
    p.set_defaults(**config_vars)
    args = p.parse_args()

    n_devices = args.n_devices or jax.local_device_count()
    print(f"jax devices ({n_devices} used of {jax.local_device_count()} local): {jax.devices()}")

    cfg = Config(**{k: getattr(args, k) for k in CONFIG_FIELDS})

    (train_np, train_labels), (val_np, val_labels) = load_cifar10(Path(args.data_root))
    if args.train_subset_n:
        train_np, train_labels = train_np[:args.train_subset_n], train_labels[:args.train_subset_n]
    train_iter = BatchIterator(train_np, train_labels, args.batch_size, n_devices, shuffle=True, seed=args.seed)
    val_iter = BatchIterator(val_np, val_labels, args.batch_size, n_devices, shuffle=False, seed=args.seed + 1)

    rng = jax.random.PRNGKey(args.seed)
    model = CausalAttnModel(rng, cfg)
    n_params = count_params(model)

    lr_schedule = warmup_schedule(args.lr, args.warmup_steps)
    optimizer = optax.adamw(lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    start_epoch = 1
    step = 0
    if args.resume_from:
        model, opt_state, step, start_epoch = load_checkpoint(Path(args.resume_from), model, opt_state)
        start_epoch += 1

    def replicate(pytree):
        return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_devices,) + x.shape)
                                       if eqx.is_array(x) else x, pytree)

    p_model = replicate(model)
    p_opt_state = replicate(opt_state)

    train_step = make_train_step(optimizer)
    eval_step = make_eval_step()

    run_dir = MODULE_DIR / "logs" / args.run_name
    logger = Logger(run_dir)
    write_resolved_config(run_dir, args)
    (run_dir / f"config_{args.config.name}").write_text(args.config.read_text())
    logger(f"config: {asdict(cfg)}")
    logger(f"run args: epochs={args.epochs} lr={args.lr} warmup_steps={args.warmup_steps} "
           f"batch_size={args.batch_size} n_devices={n_devices} resume_from={args.resume_from}")
    logger(f"params: {n_params / 1e6:.2f}M, devices={jax.devices()}")

    def run_eval() -> float:
        bpbs, accs = [], []
        for i, (r, g, b, y) in enumerate(val_iter):
            bpb, acc, u0, u1, u2 = eval_step(p_model, r, g, b, y)
            bpbs.append(float(bpb[0]))
            accs.append(float(acc[0]))
            if i >= 20:
                break
        bpb, acc = sum(bpbs) / len(bpbs), sum(accs) / len(accs)
        logger(f"val bpb={bpb:.4f} acc={acc:.4f}", val_bpb=bpb, val_acc=acc)
        return bpb

    train_prompt = train_np[:args.qual_gen_n]
    val_prompt = val_np[:args.qual_gen_n]

    def run_qual_gen(epoch: int) -> None:
        single_model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_model)
        gkw = dict(greedy=args.qual_gen_greedy, temperature=args.qual_gen_temperature, seed=epoch)

        modes = {
            "free": dict(n_prompt=0),
            "trainprompt": dict(n_prompt=1, full_prompt_r=jnp.array(train_prompt[..., 0]),
                                 full_prompt_g=jnp.array(train_prompt[..., 1]), full_prompt_b=jnp.array(train_prompt[..., 2])),
            "valprompt": dict(n_prompt=1, full_prompt_r=jnp.array(val_prompt[..., 0]),
                               full_prompt_g=jnp.array(val_prompt[..., 1]), full_prompt_b=jnp.array(val_prompt[..., 2])),
        }
        for mode_name, extra in modes.items():
            samples = single_model.generate(args.qual_gen_n, **gkw, **extra)
            out_path = run_dir / f"samples_epoch{epoch}_{mode_name}.png"
            save_sample_grid(np.asarray(samples), out_path)
            if mode_name == "trainprompt":
                save_compare_grid(np.asarray(samples), np.asarray(train_prompt),
                                   run_dir / f"samples_epoch{epoch}_traincompare.png")
        logger(f"saved qual-gen samples (free/trainprompt/valprompt/traincompare) for epoch {epoch}")

    def run_checkpoint(epoch: int) -> None:
        single_model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_model)
        single_opt_state = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, p_opt_state)
        ckpt_dir = run_dir / "checkpoints" / f"epoch_{epoch}"
        save_checkpoint(ckpt_dir, single_model, single_opt_state, step, epoch)
        logger(f"saved checkpoint at epoch {epoch} -> {ckpt_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        pbar = tqdm(train_iter, desc=f"epoch {epoch}/{args.epochs}")
        for r, g, b, y in pbar:
            p_model, p_opt_state, aux = train_step(p_model, p_opt_state, r, g, b, y)
            bpb, acc, u0, u1, u2 = aux
            step += 1
            if step % args.log_every == 0:
                logger(f"epoch={epoch} step={step} bpb={float(bpb[0]):.4f} acc={float(acc[0]):.4f} "
                       f"util(l0/l1/l2)={float(u0[0]):.2f}/{float(u1[0]):.2f}/{float(u2[0]):.2f}",
                       epoch=epoch, step=step, train_bpb=float(bpb[0]), train_acc=float(acc[0]))
        pbar.close()

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            run_eval()
            run_qual_gen(epoch)
        if epoch % args.checkpoint_every_epochs == 0 or epoch == args.epochs:
            run_checkpoint(epoch)

    logger("training done")


if __name__ == "__main__":
    main()
