import argparse
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass, fields as dataclass_fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Logger:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.text_path = run_dir / "run.log"
        self.json_path = run_dir / "run.jsonl"
        self.text_f = open(self.text_path, "a")
        self.json_f = open(self.json_path, "a")
        self.start_time = time.time()

    def __call__(self, msg: str, **record) -> None:
        elapsed_s = int(time.time() - self.start_time)
        elapsed_hms = format_hms(elapsed_s)
        line = f"[{elapsed_hms}] {msg}"
        tqdm.write(line)
        self.text_f.write(line + "\n")
        self.text_f.flush()
        json_record = {"elapsed_s": elapsed_s, "elapsed_hms": elapsed_hms, **({} if record else {"msg": msg}), **record}
        self.json_f.write(json.dumps(json_record) + "\n")
        self.json_f.flush()


class Checkpointer:
    def __init__(self, run_dir: Path, save_every_n_evals: int = 1, minimize: bool = True):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = run_dir / "best.pt"
        self.last_path = run_dir / "last.pt"
        self.save_every_n_evals = max(1, save_every_n_evals)
        self.minimize = minimize
        self.best_metric = float("inf") if minimize else float("-inf")
        self._eval_count = 0

    def is_better(self, metric: float) -> bool:
        return metric < self.best_metric if self.minimize else metric > self.best_metric

    def step(self, state: dict, metric: float) -> None:
        self._eval_count += 1
        if self.is_better(metric):
            self.best_metric = metric
            torch.save(state, self.best_path)
        if self._eval_count % self.save_every_n_evals == 0:
            torch.save(state, self.last_path)


@dataclass
class Config:
    Ks: tuple[int, ...] = (32, 32)
    d_model: int = 256
    n_layers: int = 2
    context_len: int = 1024
    n_heads: int = 4
    mlp_mult: int = 4
    attn_window: int | tuple[int, ...] = 32
    rope_base: float = 10000.0
    byte_ntp_weight: float = 1.0
    code_ntp_weight: float = 1.0
    decode_ntp_weight: float = 1.0
    gumbel_tau: float = 1.0
    use_gumbel_noise: bool = False
    vocab: int = 256
    code_extract_mode: str = "last_h"  # "last_h" | "softmax_pool" | "light_query_attn" | "query_embed"
    code_head_tied: bool = False
    decode_pack_mode: str = "interleave"  # "interleave" | "prepend"
    decode_chunked: bool = False  # use windowed chunked decode attention instead of dense O((2L)^2); interleave only
    decode_banded: bool = False  # general O(L*sum(windows)) alternative to dense O((2L)^2), any track count/K/mode
    # (see LevelLM._packed_decode_forward_banded) -- exact row-wise mask, just minimal in extent. Falls back to
    # dense automatically if any active track has an unbounded window (None), since there's no sub-quadratic
    # algorithm for that case. Dense (_packed_decode_forward) remains the reference impl; keep this False unless
    # verified to match it for your config (see scripts/test_v4_4_banded_decode.py).
    cross_track_source: str = "encode"  # "encode" | "decode" -- where a CROSS track (a coarser level
    # j>i's code, used as conditioning input for level i's decode) is sourced from. "encode" (default,
    # original behavior): level j's plain uncond encode pass (c_list[j], self-attention only, no code
    # conditioning) -- a single well-defined quantity computed once, used everywhere consistently.
    # "decode": level j's OWN cond (self-conditioned) decode pass instead -- LevelLM.forward already
    # computes a fresh code from decode's packed/conditioned hidden state (same pooling+classify+
    # quantize pipeline as encode, just fed decode's h instead) and returns it as its first value;
    # previously always discarded by _run's decode loop. Rationale (user-proposed): decode is
    # reconstruction-from-latent (code -> embed -> reconstruct), so recursively taking a lower level's
    # cross-track input from the level above's OWN reconstruction-style decode is arguably more
    # consistent with the architecture's generative structure than pulling from an uncond-NTP-focused
    # encode pass. Requires TOP-DOWN decode order (level n_levels-1 decodes first, ..., level 0 last)
    # so each level's decode-derived code is available before the level below needs it -- _run always
    # iterates top-down now (harmless for "encode", since c_list is fully built before decode starts
    # either way). SELF tracks (a level conditioning on its OWN code) always use c_list[i] regardless
    # of this setting -- a level can't condition its own decode on its own not-yet-decoded output.
    # Degenerate n_levels==1 has no cross track at all -- this flag has NO effect there, must use
    # encode's code directly (no higher-level decode exists to source from).
    decode_self_only_aux: bool = False  # without this, level i's decode is trained on exactly ONE fixed track
    # combination every step (self + every coarser level with a nonzero window) -- "self-only" (self track,
    # coarser tracks dropped, i.e. decode trained as if this were a plain n_levels==1 self-conditioning config)
    # never gets ANY gradient signal, even though ragged-length generation steps (and a collapsed/unreliable
    # coarser level's own AR codes) can put decode in exactly that regime at inference time. When True, an
    # ADDITIONAL decode forward pass runs every step using ONLY the self track (tracks[:1]), unconditionally (not
    # instead of the full-cumulative pass -- both run, every step; this is a second always-on loss term, not a
    # random dropout/mode switch). Its loss is weighted by decode_self_only_weight and added to the total loss;
    # decode_losses[i] (the full-cumulative pass, used for byte_loss/val_bpb reporting) is untouched. No effect
    # for levels with only one active track (nothing to isolate).
    decode_self_only_weight: float = 1.0  # weight on the decode_self_only_aux loss term, if enabled
    decode_code_ste: bool = True  # straight-through: decode's code_kv is source_c @ embed.weight (gradient flows into
    # the code producer). False = source_c.detach() @ embed.weight (hard argmax-equivalent forward value, no gradient
    # into the code producer -- the original behavior before this flag existed). Forward VALUE is identical either
    # way (source_c's forward value is already the hard one-hot); only the backward path differs. See docs/
    # two_stage_latent_decode_math.md -- False is REQUIRED for the drafted-substitution generation scheme there.
    # (A drafter for an n_levels==1 config is just Ks=(K0, 1) -- reuses the existing shared-weight
    # LevelLM/generate_level1_codes machinery, no separate module needed; see two_stage_latent_decode_math.md.)
    share_level_weights: bool = True  # True (default, original behavior): ONE shared LevelLM instance is used
    # for every level's encode pass AND decode pass -- literally the same object, same embed/blocks/ln_f/code_head
    # weights everywhere. False: every level gets its own INDEPENDENT encode LM and its own INDEPENDENT decode LM
    # (2*n_levels separate LevelLM instances total, e.g. n_levels==1 gives one "encode_lm" doing uncond NTP and one
    # separate "decode_lm" doing code-conditioned NTP, sharing NO weights -- not embed, not transformer blocks, not
    # head). The ONLY thing crossing from encode to decode (or between levels) in that case is the bare integer code
    # id (source_c.argmax(-1)) -- decode always re-embeds that id in ITS OWN embedding table
    # (self.decode_lms[i].embed), never touching whichever LM produced/represented it. See RefineLM.__init__ and
    # RefineLM._run's code_embeds construction.


def gumbel_quantize(logits: torch.Tensor, tau: float, use_gumbel_noise: bool = False) -> torch.Tensor:
    if use_gumbel_noise:
        eps = torch.finfo(logits.dtype).tiny
        u = torch.rand_like(logits).clamp(min=eps, max=1.0 - eps)
        gumbel_noise = -torch.log(-torch.log(u))
        soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)
    else:
        soft = F.softmax(logits / tau, dim=-1)
    hard = F.one_hot(soft.argmax(-1), num_classes=logits.shape[-1]).to(soft.dtype)
    return soft + (hard - soft).detach()


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rope_cos_sin_for_positions(position_ids: torch.Tensor, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(position_ids.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self._warned_dense_fallback = False
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None):
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if window is not None and T % window == 0 and T > window:
            y = self._forward_chunked(q, k, v, window)
        else:
            if window is not None and not self._warned_dense_fallback:
                print(f"WARNING: CausalSelfAttention window={window} set but T={T} doesn't satisfy "
                      f"T % window == 0 and T > window -- falling back to DENSE attention for this layer.")
                self._warned_dense_fallback = True
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))

    def _chunk_local_window(self, t: torch.Tensor, B: int, H: int, n_chunks: int, W: int, hd: int) -> torch.Tensor:
        tc = t.view(B, H, n_chunks, W, hd)
        zero_chunk = torch.zeros(B, H, 1, W, hd, device=t.device, dtype=t.dtype)
        t_prev = torch.cat([zero_chunk, tc[:, :, :-1]], dim=2)
        t_local = torch.cat([t_prev, tc], dim=3)
        return t_local.permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, 2 * W, hd)

    def _causal_window_mask(self, n_chunks: int, W: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(W, device=device).view(W, 1)
        j_prev = torch.arange(W, device=device).view(1, W) - W
        j_cur = torch.arange(W, device=device).view(1, W)
        key_offset = torch.cat([j_prev, j_cur], dim=1)
        diff = i - key_offset
        causal_window = (diff >= 0) & (diff < W)
        mask_per_chunk = causal_window.unsqueeze(0).expand(n_chunks, W, 2 * W).clone()
        mask_per_chunk[0, :, 0:W] = False
        return mask_per_chunk

    def _forward_chunked(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
        B, H, T, hd = q.shape
        W = window
        n_chunks = T // W

        qb = q.view(B, H, n_chunks, W, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, W, hd)
        kb = self._chunk_local_window(k, B, H, n_chunks, W, hd)
        vb = self._chunk_local_window(v, B, H, n_chunks, W, hd)
        mask_per_chunk = self._causal_window_mask(n_chunks, W, q.device)
        mask_batched = mask_per_chunk.unsqueeze(0).expand(B, n_chunks, W, 2 * W).reshape(B * n_chunks, 1, W, 2 * W)

        yb = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask_batched)
        return yb.view(B, n_chunks, H, W, hd).permute(0, 2, 1, 3, 4).reshape(B, H, T, hd)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, window: int | None):
        a = self.attn(self.ln1(x), cos, sin, window)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class LevelLM(nn.Module):
    """Pure weight-holder -- no level-specific state (level index, window, decode_windows) stored
    on the instance anymore. Those are passed as forward() arguments instead, so the SAME instance
    can safely be reused (literally aliased) across multiple levels/roles when Config.
    share_level_weights=True (see RefineLM.__init__), without any stale per-level state leaking
    across levels. Always constructs its own full parameter set at construction time -- sharing
    (or not) is handled entirely by RefineLM choosing whether to construct one LevelLM and alias
    it everywhere, or construct 2*n_levels independent ones."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        V = cfg.vocab
        self.embed = nn.Embedding(V, D)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(D)
        self.code_head = None if cfg.code_head_tied else nn.Linear(D, V, bias=False)
        if self.code_head is not None:
            nn.init.normal_(self.code_head.weight, std=0.02)
        self.code_query = self.code_out = self.query_embed = None
        if cfg.code_extract_mode == "light_query_attn":
            self.code_query = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.code_query, std=0.02)
            self.code_out = nn.Linear(D, D, bias=False)
        elif cfg.code_extract_mode == "query_embed":
            self.query_embed = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.query_embed, std=0.02)
        self.decode_bos = nn.Parameter(torch.zeros(D))
        nn.init.normal_(self.decode_bos, std=0.02)

    def _classify(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.code_head(pooled) if self.code_head is not None else F.linear(pooled, self.embed.weight)

    def _query_embed_pool(self, x0: torch.Tensor, K: int, n_blocks: int, window: int | None) -> torch.Tensor:
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device

        x0_blocks = x0.view(B, n_blocks, K, D)
        q_tok = self.query_embed.view(1, 1, 1, D).expand(B, n_blocks, 1, D)
        xe = torch.cat([x0_blocks, q_tok], dim=2).view(B, n_blocks * (K + 1), D)
        Le = n_blocks * (K + 1)

        slot = torch.arange(K + 1, device=device).repeat(n_blocks)
        is_query = slot == K
        block_of = torch.arange(n_blocks, device=device).repeat_interleave(K + 1)
        within_block_pos = torch.where(is_query, torch.full_like(slot, K - 1), slot)
        real_pos = block_of * K + within_block_pos

        cos, sin = rope_cos_sin_for_positions(real_pos, hd, cfg.rope_base, device)

        win = window if window is not None else L
        ti = real_pos.unsqueeze(1)
        tj = real_pos.unsqueeze(0)
        causal = tj <= ti
        windowed = (ti - tj) < win
        query_only_sees_own_block = (~is_query).unsqueeze(0) | (block_of.unsqueeze(1) == block_of.unsqueeze(0))
        real_never_sees_query = ~(is_query.unsqueeze(0) & (~is_query).unsqueeze(1))
        allow = causal & windowed & query_only_sees_own_block & real_never_sees_query
        attn_mask = allow.view(1, 1, Le, Le)

        for block in self.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, Le, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            a = block.attn.out(y.transpose(1, 2).reshape(B, Le, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))

        he = self.ln_f(xe)
        he_blocks = he.view(B, n_blocks, K + 1, D)
        return he_blocks[:, :, K, :]

    def _packed_decode_forward(self, x0: torch.Tensor, tracks: list[tuple[torch.Tensor, int, int | None]]) -> torch.Tensor:
        """tracks: one (code_kv, K, window) triple per decode source, ordered [self, +1 level,
        +2 levels, ..., top level] (finest first -- matches RefineLM._run's own source-gathering
        order). Generalizes the original single-source decode_K==1 mechanism to CUMULATIVE
        multi-level conditioning: level i's decode sees its own code (K=Ks[i]) AND every coarser
        level's code above it (K=Ks[i]*Ks[i+1], Ks[i]*Ks[i+1]*Ks[i+2], ...), each with its own
        window. Every track uses the same prefix-per-block mechanism as before (one prefix token
        per K-byte block, prefix b is decode_bos for b==0 else code_kv[b-1], true_pos = b*K - 1)
        -- what's new is packing MULTIPLE such prefix streams together.

        Single track (len(tracks)==1) still supports decode_pack_mode=="interleave" (the
        original, denser-looking one-prefix-per-byte layout at K==1, needed by the chunked decode
        path). Multiple tracks only support "prepend": all prefix streams concatenated
        coarsest-to-finest, self (finest) track immediately before the bytes -- order doesn't
        affect correctness (causality is governed by true_pos values, not physical position), but
        coarsest-first-then-narrower-then-bytes is the more readable convention.
        """
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device
        byte_pos = torch.arange(L, device=device)

        if len(tracks) == 1 and cfg.decode_pack_mode == "interleave":
            code_kv, K, window = tracks[0]
            W = window if window is not None else L
            assert L % K == 0
            n_blocks = L // K
            bos = self.decode_bos.view(1, 1, D).expand(B, 1, D)
            prefixes = torch.cat([bos, code_kv[:, :-1, :]], dim=1)  # [B, n_blocks, D]
            prefix_true_pos = torch.arange(n_blocks, device=device) * K - 1
            x0_blocks = x0.view(B, n_blocks, K, D)
            combined = torch.cat([prefixes.unsqueeze(2), x0_blocks], dim=2).view(B, n_blocks * (K + 1), D)
            true_pos = torch.cat([prefix_true_pos.view(n_blocks, 1), byte_pos.view(n_blocks, K)], dim=1).reshape(-1)
            is_code = torch.cat([torch.ones(n_blocks, 1, dtype=torch.bool, device=device),
                                  torch.zeros(n_blocks, K, dtype=torch.bool, device=device)], dim=1).reshape(-1)
            window_of_key = torch.cat([torch.full((n_blocks, 1), float(W), device=device),
                                        torch.full((n_blocks, K), float(W), device=device)], dim=1).reshape(-1)
            Le = n_blocks * (K + 1)
            extract = lambda he: he.view(B, n_blocks, K + 1, D)[:, :, 1:, :].reshape(B, L, D)
        elif cfg.decode_pack_mode in ("interleave", "prepend"):
            prefix_parts, true_pos_parts, window_parts = [], [], []
            for code_kv, K, window in tracks:
                assert L % K == 0
                n_blocks = L // K
                W = window if window is not None else L
                bos = self.decode_bos.view(1, 1, D).expand(B, 1, D)
                prefixes = torch.cat([bos, code_kv[:, :-1, :]], dim=1)  # [B, n_blocks, D]
                prefix_parts.append(prefixes)
                true_pos_parts.append(torch.arange(n_blocks, device=device) * K - 1)
                window_parts.append(torch.full((n_blocks,), float(W), device=device))
            prefix_parts, true_pos_parts, window_parts = prefix_parts[::-1], true_pos_parts[::-1], window_parts[::-1]

            all_prefixes = torch.cat(prefix_parts, dim=1)  # [B, n_prefix, D]
            n_prefix = all_prefixes.shape[1]
            prefix_true_pos = torch.cat(true_pos_parts, dim=0)
            prefix_window = torch.cat(window_parts, dim=0)
            byte_window = tracks[0][2] if tracks[0][2] is not None else L  # self track's window governs the bytes

            combined = torch.cat([all_prefixes, x0], dim=1)
            true_pos = torch.cat([prefix_true_pos, byte_pos], dim=0)
            is_code = torch.cat([torch.ones(n_prefix, dtype=torch.bool, device=device),
                                  torch.zeros(L, dtype=torch.bool, device=device)])
            window_of_key = torch.cat([prefix_window, torch.full((L,), float(byte_window), device=device)])
            Le = n_prefix + L
            extract = lambda he: he[:, n_prefix:, :]
        else:
            raise ValueError(f"unknown decode_pack_mode {cfg.decode_pack_mode!r}")

        cos, sin = rope_cos_sin_for_positions(true_pos.clamp(min=0), hd, cfg.rope_base, device)

        ti = true_pos.unsqueeze(1)
        tj = true_pos.unsqueeze(0)
        key_is_code = is_code.unsqueeze(0)
        causal = tj <= ti
        same_pos_code_excluded = ~(key_is_code & (tj == ti))
        windowed = (ti - tj) < (2 * window_of_key.unsqueeze(0))
        allow = causal & same_pos_code_excluded & windowed
        attn_mask = allow.view(1, 1, Le, Le)

        xe = combined
        for block in self.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, Le, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            a = block.attn.out(y.transpose(1, 2).reshape(B, Le, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))

        he = self.ln_f(xe)
        return extract(he)

    def _packed_decode_forward_banded(self, x0: torch.Tensor, tracks: list[tuple[torch.Tensor, int, int | None]],
                                       margin_extra_chunks: int = 1) -> torch.Tensor:
        """General O(L * (sum of windows)) alternative to _packed_decode_forward's O((2L)^2) dense
        attention -- exact row-wise mask, same causal/same_pos_code_excluded/windowed formula, just
        never materialized as a dense Le x Le matrix. Works for ANY number of tracks, ANY per-track
        K/window, regardless of cfg.decode_pack_mode (see below).

        Key idea: _packed_decode_forward's own docstring already notes physical packing order
        doesn't affect correctness -- only true_pos does. So here, instead of choosing a packing
        order and hoping it stays roughly true_pos-monotonic (interleave, K==1 only) or accepting
        it isn't (prepend, multi-track) and paying dense attention for it, we build the sequence
        once (prepend-style: all prefixes then bytes -- picked arbitrarily, discarded after sorting)
        and explicitly SORT by true_pos. That makes the sequence monotonic unconditionally, so the
        same chunk-with-margin banding trick _packed_decode_forward_chunked already uses for the
        K==1 interleave case generalizes directly, with one difference: window is now PER-KEY
        (window_of_key), not a single shared scalar, since different tracks can have different
        windows. The `allow` mask formula is copied verbatim from the dense version, just evaluated
        on a small gathered (sc, Kc) window per query-chunk instead of the full (Le, Le) matrix --
        still an exact row-wise mask, just minimal in extent rather than absent.

        Falls back to being a no-op speedup (n_prev_chunks -> covers everything) when any window is
        unbounded (None, i.e. full context) -- there's no sub-quadratic algorithm for an unbounded
        window, same as CausalSelfAttention's own dense fallback when window==T. Caller should check
        for that case (see forward()'s dispatch) and prefer the dense reference there instead.
        """
        cfg = self.cfg
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device
        byte_pos = torch.arange(L, device=device)

        prefix_parts, true_pos_parts, window_parts = [], [], []
        for code_kv, K, window in tracks:
            assert L % K == 0
            n_blocks = L // K
            W = window if window is not None else L
            bos = self.decode_bos.view(1, 1, D).expand(B, 1, D)
            prefixes = torch.cat([bos, code_kv[:, :-1, :]], dim=1)
            prefix_parts.append(prefixes)
            true_pos_parts.append(torch.arange(n_blocks, device=device) * K - 1)
            window_parts.append(torch.full((n_blocks,), float(W), device=device))
        prefix_parts, true_pos_parts, window_parts = prefix_parts[::-1], true_pos_parts[::-1], window_parts[::-1]

        all_prefixes = torch.cat(prefix_parts, dim=1)
        n_prefix = all_prefixes.shape[1]
        prefix_true_pos = torch.cat(true_pos_parts, dim=0)
        prefix_window = torch.cat(window_parts, dim=0)
        byte_window = tracks[0][2] if tracks[0][2] is not None else L

        combined = torch.cat([all_prefixes, x0], dim=1)
        true_pos = torch.cat([prefix_true_pos, byte_pos], dim=0)
        is_code = torch.cat([torch.ones(n_prefix, dtype=torch.bool, device=device),
                              torch.zeros(L, dtype=torch.bool, device=device)])
        window_of_key = torch.cat([prefix_window, torch.full((L,), float(byte_window), device=device)])
        Le = n_prefix + L

        # Sort by true_pos, with ties (code and byte sharing one true_pos -- a track's code sits
        # right at its block boundary) broken bytes-before-codes. This matters because the mask is
        # ASYMMETRIC at ties: a code query CAN see a same-true_pos byte key (same_pos_code_excluded
        # only excludes CODE keys), but a byte query CANNOT see a same-true_pos code key. A
        # backward-only gather over one sorted sequence can only realize that asymmetry if the
        # "can't be seen" side (codes, at code-vs-code ties symmetric exclusion makes order among
        # codes themselves irrelevant) sorts AFTER the "can be seen" side (bytes) -- so codes can
        # look back and find same-true_pos bytes, while bytes never look far enough forward to find
        # same-true_pos codes in the first place.
        sort_key = true_pos * 2 + is_code.long()
        perm = torch.argsort(sort_key)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(Le, device=device)

        true_pos_s = true_pos[perm]
        is_code_s = is_code[perm]
        window_of_key_s = window_of_key[perm]
        combined_s = combined[:, perm, :]

        W_max = float(window_of_key.max().item())
        R = 2.0 * W_max

        sc = max(1, int(tracks[0][1]))  # chunk size = finest (self) track's K
        n_chunks = -(-Le // sc)
        pad_len = n_chunks * sc - Le
        if pad_len > 0:
            combined_s = F.pad(combined_s, (0, 0, 0, pad_len))
            true_pos_s = F.pad(true_pos_s, (0, pad_len), value=1e9)  # sentinel: never a valid key or usable query
            is_code_s = F.pad(is_code_s, (0, pad_len), value=False)
            window_of_key_s = F.pad(window_of_key_s, (0, pad_len), value=0.0)
        Lp = n_chunks * sc

        # Sorted true_pos isn't strictly increasing per step -- codes tie with bytes at the same
        # true_pos (a track's code sits right before its block, one unit below the boundary byte),
        # and up to len(tracks)+1 entries (one per track's code, plus the byte stream) can share
        # one true_pos value. So a run of that many sorted-array steps can advance true_pos by as
        # little as 0; covering reach R therefore needs up to (len(tracks)+1)*R sorted STEPS, not R.
        max_ties_per_pos = len(tracks) + 1
        n_prev_chunks = max(1, int(math.ceil(max_ties_per_pos * R / sc)) + margin_extra_chunks)

        cos, sin = rope_cos_sin_for_positions(true_pos_s.clamp(min=0), hd, cfg.rope_base, device)

        pos_b = true_pos_s.view(n_chunks, sc)
        code_b = is_code_s.view(n_chunks, sc)
        win_b = window_of_key_s.view(n_chunks, sc)
        pad_pos = torch.full((n_prev_chunks, sc), -1e9, device=device, dtype=pos_b.dtype)
        pad_code = torch.zeros(n_prev_chunks, sc, dtype=torch.bool, device=device)
        pad_win = torch.zeros(n_prev_chunks, sc, device=device, dtype=win_b.dtype)
        pos_ext = torch.cat([pad_pos, pos_b], dim=0)
        code_ext = torch.cat([pad_code, code_b], dim=0)
        win_ext = torch.cat([pad_win, win_b], dim=0)

        idx = (torch.arange(n_chunks, device=device).view(n_chunks, 1)
               + torch.arange(n_prev_chunks + 1, device=device).view(1, n_prev_chunks + 1))
        Kc = (n_prev_chunks + 1) * sc
        pos_win = pos_ext[idx].reshape(n_chunks, Kc)
        code_win = code_ext[idx].reshape(n_chunks, Kc)
        win_win = win_ext[idx].reshape(n_chunks, Kc)

        ti = pos_b.unsqueeze(-1)
        tj = pos_win.unsqueeze(1)
        key_is_code = code_win.unsqueeze(1)
        key_window = win_win.unsqueeze(1)
        causal = tj <= ti
        same_pos_excl = ~(key_is_code & (tj == ti))
        windowed = (ti - tj) < (2.0 * key_window)
        allow = causal & same_pos_excl & windowed  # [n_chunks, sc, Kc], exact same formula as dense
        attn_mask = allow.view(1, n_chunks, 1, sc, Kc)

        xe = combined_s
        for block in self.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, Lp, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

            qb = q.view(B, H, n_chunks, sc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, sc, hd)
            kb_flat = k.view(B, H, n_chunks, sc, hd)
            vb_flat = v.view(B, H, n_chunks, sc, hd)
            pad_k = torch.zeros(B, H, n_prev_chunks, sc, hd, device=device, dtype=k.dtype)
            pad_v = torch.zeros(B, H, n_prev_chunks, sc, hd, device=device, dtype=v.dtype)
            k_ext = torch.cat([pad_k, kb_flat], dim=2)
            v_ext = torch.cat([pad_v, vb_flat], dim=2)
            k_win = k_ext[:, :, idx].reshape(B, H, n_chunks, Kc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Kc, hd)
            v_win = v_ext[:, :, idx].reshape(B, H, n_chunks, Kc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Kc, hd)

            mask_batched = attn_mask.expand(B, n_chunks, 1, sc, Kc).reshape(B * n_chunks, 1, sc, Kc)
            yb = F.scaled_dot_product_attention(qb, k_win, v_win, attn_mask=mask_batched)
            y = yb.view(B, n_chunks, H, sc, hd).permute(0, 2, 1, 3, 4).reshape(B, H, Lp, hd)

            a = block.attn.out(y.transpose(1, 2).reshape(B, Lp, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))

        he = self.ln_f(xe)
        he = he[:, :Le, :]       # drop chunk-fill padding (physically at the tail, post-sort)
        he = he[:, inv_perm, :]  # restore original (prefixes-then-bytes) order
        return he[:, n_prefix:, :]

    def _packed_decode_forward_chunked(self, x0: torch.Tensor, code_kv: torch.Tensor, window: int, n_prev_chunks: int = 2) -> torch.Tensor:
        """Chunked equivalent of _packed_decode_forward for decode_pack_mode == "interleave" only,
        single-track (decode_K==1) only.

        Interleave's combined sequence (code_0,byte_0,code_1,byte_1,...) has true_pos
        non-decreasing in sequence order, so it can be chunked contiguously like plain windowed
        self-attention. Each byte occupies 2 slots (itself + its paired code), so a query chunk of
        sc=2*W slots is used, with n_prev_chunks previous chunks kept as extra key context to
        cover the required true-position reach R=2*W (margin empirically verified, see
        scripts/test_v4_4_chunked_decode.py).
        """
        cfg = self.cfg
        assert cfg.decode_pack_mode == "interleave", "chunked decode only implemented for interleave"
        B, L, D = x0.shape
        H, hd = cfg.n_heads, D // cfg.n_heads
        device = x0.device
        W = window
        assert W is not None and L % W == 0

        bos = self.decode_bos.view(1, 1, D).expand(B, 1, D)
        code_shifted = torch.cat([bos, code_kv[:, :-1, :]], dim=1)

        byte_pos = torch.arange(L, device=device)
        code_true_pos = byte_pos - 1

        combined = torch.stack([code_shifted, x0], dim=2).view(B, 2 * L, D)
        true_pos = torch.stack([code_true_pos, byte_pos], dim=1).reshape(-1)
        is_code = torch.tensor([True, False], device=device).repeat(L)

        Le = 2 * L
        cos, sin = rope_cos_sin_for_positions(true_pos.clamp(min=0), hd, cfg.rope_base, device)

        sc = 2 * W
        n_chunks = Le // sc
        R = 2 * W

        pos_b = true_pos.view(n_chunks, sc)
        code_b = is_code.view(n_chunks, sc)
        pad_pos = torch.full((n_prev_chunks, sc), -10 ** 9, device=device, dtype=pos_b.dtype)
        pad_code = torch.zeros(n_prev_chunks, sc, dtype=torch.bool, device=device)
        pos_ext = torch.cat([pad_pos, pos_b], dim=0)
        code_ext = torch.cat([pad_code, code_b], dim=0)

        idx = (torch.arange(n_chunks, device=device).view(n_chunks, 1)
               + torch.arange(n_prev_chunks + 1, device=device).view(1, n_prev_chunks + 1))
        pos_win = pos_ext[idx].reshape(n_chunks, (n_prev_chunks + 1) * sc)
        code_win = code_ext[idx].reshape(n_chunks, (n_prev_chunks + 1) * sc)

        ti = pos_b.unsqueeze(-1)
        tj = pos_win.unsqueeze(1)
        key_is_code = code_win.unsqueeze(1)
        causal = tj <= ti
        same_pos_excl = ~(key_is_code & (tj == ti))
        windowed = (ti - tj) < R
        allow = causal & same_pos_excl & windowed  # [n_chunks, sc, Kc]
        Kc = (n_prev_chunks + 1) * sc
        attn_mask = allow.view(1, n_chunks, 1, sc, Kc)

        xe = combined
        for block in self.blocks:
            xn = block.ln1(xe)
            qkv = block.attn.qkv(xn).reshape(B, Le, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

            qb = q.view(B, H, n_chunks, sc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, sc, hd)
            kb_flat = k.view(B, H, n_chunks, sc, hd)
            vb_flat = v.view(B, H, n_chunks, sc, hd)
            pad_k = torch.zeros(B, H, n_prev_chunks, sc, hd, device=device, dtype=k.dtype)
            pad_v = torch.zeros(B, H, n_prev_chunks, sc, hd, device=device, dtype=v.dtype)
            k_ext = torch.cat([pad_k, kb_flat], dim=2)
            v_ext = torch.cat([pad_v, vb_flat], dim=2)
            k_win = k_ext[:, :, idx].reshape(B, H, n_chunks, Kc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Kc, hd)
            v_win = v_ext[:, :, idx].reshape(B, H, n_chunks, Kc, hd).permute(0, 2, 1, 3, 4).reshape(B * n_chunks, H, Kc, hd)

            mask_batched = attn_mask.expand(B, n_chunks, 1, sc, Kc).reshape(B * n_chunks, 1, sc, Kc)
            yb = F.scaled_dot_product_attention(qb, k_win, v_win, attn_mask=mask_batched)
            y = yb.view(B, n_chunks, H, sc, hd).permute(0, 2, 1, 3, 4).reshape(B, H, Le, hd)

            a = block.attn.out(y.transpose(1, 2).reshape(B, Le, D))
            xe = xe + a
            xe = xe + block.mlp(block.ln2(xe))

        he = self.ln_f(xe)
        he_pairs = he.view(B, L, 2, D)
        return he_pairs[:, :, 1, :]

    def forward(self, seq_repr: torch.Tensor, level: int, window: int | None, compute_ntp: bool = True,
                decode_tracks: list[tuple[torch.Tensor, int, int | None]] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        K = cfg.Ks[level]
        D = cfg.d_model
        is_byte_level = level == 0

        if is_byte_level:
            x = self.embed(seq_repr)
            B, L = seq_repr.shape
        else:
            x = seq_repr @ self.embed.weight
            B, L, _ = seq_repr.shape

        x0 = x
        head_dim = D // cfg.n_heads

        if decode_tracks is not None:
            assert len(decode_tracks) >= 1
            _, k0, w0 = decode_tracks[0]
            can_chunk = (len(decode_tracks) == 1 and k0 == 1 and cfg.decode_chunked
                         and cfg.decode_pack_mode == "interleave" and w0 is not None and L % w0 == 0)
            can_band = (cfg.decode_banded and all(w is not None and w < L for _, _, w in decode_tracks))
            if can_chunk:
                h = self._packed_decode_forward_chunked(x0, decode_tracks[0][0], w0)
            elif can_band:
                h = self._packed_decode_forward_banded(x0, decode_tracks)
            else:
                h = self._packed_decode_forward(x0, decode_tracks)
        else:
            cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
            for block in self.blocks:
                x = block(x, cos, sin, window)
            h = self.ln_f(x)

        if compute_ntp:
            h_flat = h[:, :-1, :].reshape(-1, D)
            target = (seq_repr[:, 1:].reshape(-1) if is_byte_level
                      else seq_repr[:, 1:, :].argmax(-1).reshape(-1))
            logits = F.linear(h_flat, self.embed.weight)
            ntp_loss = F.cross_entropy(logits, target)
            with torch.no_grad():
                ntp_acc = (logits.argmax(-1) == target).float().mean()
        else:
            ntp_loss = h.new_zeros(())
            ntp_acc = h.new_zeros(())

        n_blocks = L // K
        h_blocks = h[:, :n_blocks * K, :].view(B, n_blocks, K, D)
        if cfg.code_extract_mode == "last_h":
            pooled = h_blocks[:, :, K - 1, :]
        elif cfg.code_extract_mode == "softmax_pool":
            q_implicit = h_blocks[:, :, K - 1, :]
            scores = (h_blocks * q_implicit.unsqueeze(2)).sum(-1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1)
            pooled = (weights.unsqueeze(-1) * h_blocks).sum(2)
        elif cfg.code_extract_mode == "light_query_attn":
            scores = (h_blocks * self.code_query.view(1, 1, 1, D)).sum(-1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1)
            pooled = (weights.unsqueeze(-1) * h_blocks).sum(2)
            pooled = self.code_out(pooled)
        elif cfg.code_extract_mode == "query_embed":
            pooled = self._query_embed_pool(x0, K, n_blocks, window)
        else:
            raise ValueError(f"unknown code_extract_mode {cfg.code_extract_mode!r}")
        pre_q = self._classify(pooled)
        c_i = gumbel_quantize(pre_q, cfg.gumbel_tau, cfg.use_gumbel_noise)

        return c_i, ntp_loss, ntp_acc, h


class RefineLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_levels = len(cfg.Ks)
        assert cfg.d_model % cfg.n_heads == 0, f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"

        seq_lens = [cfg.context_len]
        for k in cfg.Ks[:-1]:
            assert seq_lens[-1] % k == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
            seq_lens.append(seq_lens[-1] // k)
        assert seq_lens[-1] % cfg.Ks[-1] == 0, f"Ks={cfg.Ks} must evenly divide context_len at every level"
        self.seq_lens = seq_lens

        # attn_window: one entry per level, each a scalar (broadcast to encode AND every decode
        # source), or an (encode_window, decode_window) 2-tuple, where decode_window is itself
        # either a scalar (broadcast to ALL of this level's decode sources -- itself plus every
        # coarser level above it, cumulative) or an explicit tuple of length (n_levels - i),
        # ordered [own/self, +1 level, +2 levels, ..., top level]. Per-source window values:
        # a positive int bounds that source's reach; -1 means unbounded/full-context (NOT
        # disabled -- the track is still included, just with no window limit); 0 means the track
        # is excluded from decode entirely (see RefineLM._run).
        raw_windows = cfg.attn_window if isinstance(cfg.attn_window, (tuple, list)) else (cfg.attn_window,) * self.n_levels
        assert len(raw_windows) == self.n_levels, f"attn_window tuple must have length n_levels={self.n_levels}, got {len(raw_windows)}"
        windows: list[int | None] = []
        decode_windows: list[list[int | None]] = []  # decode_windows[i] has length n_levels - i
        for i, w in enumerate(raw_windows):
            n_sources = self.n_levels - i
            if isinstance(w, (tuple, list)):
                assert len(w) == 2, f"attn_window[{i}] must be a scalar or an (encode_window, decode_window) 2-tuple, got {w!r}"
                ew, dw = w
            else:
                ew = dw = w
            windows.append(None if ew == -1 else ew)
            if isinstance(dw, (tuple, list)):
                assert len(dw) == n_sources, (
                    f"attn_window[{i}]'s decode_window must be a scalar (broadcast) or a tuple of "
                    f"length n_levels-{i}={n_sources} (one per decode source: self, +1, ..., top), got {dw!r}")
                decode_windows.append([None if x == -1 else x for x in dw])
            else:
                decode_windows.append([None if dw == -1 else dw] * n_sources)
        self.windows = windows
        self.decode_windows = decode_windows
        for i, (L, window) in enumerate(zip(seq_lens, windows)):
            if window is not None:
                assert L % window == 0 or L <= window, f"attn_window[{i}] encode window ({window}) must divide level {i}'s sequence length ({L}), or be >= it"
        for i, dwlist in enumerate(decode_windows):
            L = seq_lens[i]
            for src_offset, dwindow in enumerate(dwlist):
                if dwindow is not None and dwindow != 0:  # None=unbounded, 0=track disabled -- neither constrained
                    assert L % dwindow == 0 or L <= dwindow, (
                        f"attn_window[{i}]'s decode_window[{src_offset}] ({dwindow}) must divide "
                        f"level {i}'s sequence length ({L}), or be >= it")

        # encode_lms[i] / decode_lms[i]: level i's encode-pass LM and decode-pass LM. When
        # cfg.share_level_weights (default), ALL 2*n_levels slots literally alias the SAME single
        # LevelLM instance -- one shared LM does every level's encode AND decode, reproducing the
        # original (pre-this-flag) behavior exactly. When False, every slot gets its own
        # independently-constructed LevelLM -- encode and decode (and every level) have fully
        # separate embed/blocks/ln_f/code_head weights, coupled only through the bare integer code
        # ids that cross from one LM to another (see _run's code_embeds construction).
        if cfg.share_level_weights:
            shared_lm = LevelLM(cfg)
            encode_lms = [shared_lm for _ in range(self.n_levels)]
            decode_lms = [shared_lm for _ in range(self.n_levels)]
        else:
            encode_lms = [LevelLM(cfg) for _ in range(self.n_levels)]
            decode_lms = [LevelLM(cfg) for _ in range(self.n_levels)]
        self.encode_lms = nn.ModuleList(encode_lms)
        self.decode_lms = nn.ModuleList(decode_lms)

    def _run(self, byte_ids: torch.Tensor, compute_ntp: bool = True, max_decode_sources: int | None = None):
        """max_decode_sources: if set, every level's decode track list is truncated to at most this
        many sources (self=1, self+1=2, ...) before use. None (default) = no truncation, full
        cumulative set. Lets callers (generation functions) force a specific conditioning mode
        using the exact same code path training uses -- e.g. max_decode_sources=1 forces
        self-only, matching what generate_self_only_cond does and what Config.decode_self_only_aux
        trains an auxiliary loss on below."""
        cfg = self.cfg
        seq_repr = byte_ids
        encode_losses, encode_accs, h_list, c_list, x_list = [], [], [], [], []

        for i in range(self.n_levels):
            want_ntp = compute_ntp and (i == 0 or cfg.code_ntp_weight > 0)
            c_i, loss_i, acc_i, h_i = self.encode_lms[i](seq_repr, level=i, window=self.windows[i], compute_ntp=want_ntp)
            encode_losses.append(loss_i)
            encode_accs.append(acc_i)
            h_list.append(h_i)
            c_list.append(c_i)
            x_list.append(seq_repr)
            seq_repr = c_i

        # Cumulative decode: EVERY level i conditions on its own code (code_i, self) PLUS every
        # coarser level's code above it (code_{i+1}, code_{i+2}, ..., code_{n_levels-1}), each as
        # its own track with its own window (LevelLM.decode_windows[i], ordered [self, +1, +2,
        # ...]). The top level (n_levels-1) only has a self track -- same degenerate case as the
        # original n_levels==1 self-conditioning, now just the general "nothing coarser exists"
        # case for whichever level happens to be on top.
        decode_losses: list = [None] * self.n_levels
        decode_accs: list = [None] * self.n_levels
        # decode_self_only_losses/accs: an ADDITIONAL, always-on (when Config.decode_self_only_aux)
        # decode pass per level using ONLY the self track (tracks[:1]), run alongside (not instead
        # of) the full-cumulative pass above -- both contribute every step, not a random switch.
        # Gives "self-only" conditioning real gradient signal, which it otherwise never gets (see
        # Config.decode_self_only_aux's docstring). Never populated when max_decode_sources is
        # already set (that means a caller explicitly wants ONE specific mode, e.g. generation
        # forcing self-only via max_decode_sources=1 -- no separate aux pass needed there).
        decode_self_only_losses: list = [None] * self.n_levels
        decode_self_only_accs: list = [None] * self.n_levels
        # decode_derived_c[j]: level j's OWN code, re-derived from its cond (self-conditioned)
        # decode pass rather than its uncond encode pass -- LevelLM.forward computes this
        # unconditionally (same pooling+classify+quantize pipeline, just fed decode's packed/
        # conditioned h instead of encode's plain h) and returns it as its first value; only
        # captured here (instead of discarded) when cfg.cross_track_source=="decode". Populated
        # top-down as each level's decode runs, so a lower level's cross track can read a
        # coarser level's ALREADY-COMPUTED decode-derived code.
        decode_derived_c: dict[int, torch.Tensor] = {}
        h_out = list(h_list)
        for i in reversed(range(self.n_levels)):  # top-down: required for cross_track_source=="decode"
            # to see coarser levels' decode-derived codes before this level needs them; harmless
            # for "encode" (c_list is already fully built before this loop starts either way).
            L_i = x_list[i].shape[1]
            tracks: list[tuple[torch.Tensor, int, int | None]] = []
            cum_K = 1
            ragged = False
            for j in range(i, self.n_levels):
                cum_K *= cfg.Ks[j]
                window = self.decode_windows[i][j - i]
                if window == 0:
                    continue  # track disabled -- excluded, but doesn't affect coarser tracks' own cum_K
                if L_i % cum_K != 0:
                    # Ragged length (only happens during generation, where the sequence grows one
                    # byte at a time -- training always uses context_len, guaranteed divisible per
                    # RefineLM.__init__'s own asserts). Skip decode entirely for this level rather
                    # than partially conditioning on some tracks but not others.
                    ragged = True
                    break
                # SELF (j==i) always sources from encode's own code -- a level can't condition its
                # own decode on its own not-yet-decoded output. CROSS (j>i) sources from
                # cross_track_source: "decode" prefers that coarser level's decode-derived code,
                # falling back to encode's if unavailable (e.g. level j's decode was itself ragged/
                # skipped, or disabled via window==0 -- keeps this correct rather than crashing).
                if j > i and cfg.cross_track_source == "decode" and j in decode_derived_c:
                    source_c = decode_derived_c[j]
                else:
                    source_c = c_list[j]
                if cfg.decode_code_ste:
                    # STE needs the full soft matmul: source_c's VALUE is the hard one-hot
                    # (gumbel_quantize's straight-through construction), but its GRADIENT behaves
                    # as if it were the soft distribution -- that gradient estimate only exists
                    # via this matmul against embed.weight; an index lookup has no gradient w.r.t.
                    # which index was chosen, so it would silently drop the code producer's
                    # gradient entirely, defeating decode_code_ste=True's whole purpose.
                    code_embeds = source_c @ self.decode_lms[i].embed.weight
                else:
                    # detach: forward value of source_c.detach() @ embed.weight is mathematically
                    # a one-hot-row selection, IDENTICAL to embed.weight[argmax(source_c)] -- and
                    # since no gradient into source_c is wanted here anyway (that's the point of
                    # detach), a plain index lookup gives the exact same forward value AND the
                    # exact same gradient w.r.t. embed.weight (index-select's gradient scatters
                    # into the selected row, same as one_hot@W's), for less compute (no vocab x D
                    # matmul, just a gather). This is also the conceptually cleaner read of
                    # "detach": the decoder should condition on the code as a fixed discrete
                    # query/embedding lookup (latent-variable / Markov-chain style -- the decoder
                    # doesn't get to reshape which latent it's conditioning on), not as a soft
                    # mixture it could partially steer.
                    code_embeds = self.decode_lms[i].embed(source_c.argmax(-1))
                tracks.append((code_embeds, cum_K, window))
            if ragged or not tracks:
                continue
            full_tracks = tracks
            if max_decode_sources is not None:
                full_tracks = full_tracks[:max_decode_sources]
            c_i2, loss_i2, acc_i2, h_i2 = self.decode_lms[i](x_list[i], level=i, window=self.windows[i],
                                                               compute_ntp=compute_ntp, decode_tracks=full_tracks)
            decode_losses[i] = loss_i2
            decode_accs[i] = acc_i2
            h_out[i] = h_i2
            if max_decode_sources is None:
                # only the FULL (untruncated) cumulative pass's decode-derived code is stored --
                # a max_decode_sources-truncated call (generation forcing a specific mode) reflects
                # a deliberately different, narrower conditioning set, not "this level's real code".
                decode_derived_c[i] = c_i2
            if cfg.decode_self_only_aux and self.training and max_decode_sources is None and len(tracks) > 1:
                _, loss_self, acc_self, _ = self.decode_lms[i](x_list[i], level=i, window=self.windows[i],
                                                                 compute_ntp=compute_ntp, decode_tracks=tracks[:1])
                decode_self_only_losses[i] = loss_self
                decode_self_only_accs[i] = acc_self

        return (encode_losses, encode_accs, decode_losses, decode_accs, h_out, c_list,
                decode_self_only_losses, decode_self_only_accs)

    def forward(self, byte_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        (encode_losses, encode_accs, decode_losses, decode_accs, h_list, c_list,
         decode_self_only_losses, decode_self_only_accs) = self._run(byte_ids)

        byte_loss = decode_losses[0] if decode_losses[0] is not None else encode_losses[0]
        byte_acc = decode_accs[0] if decode_accs[0] is not None else encode_accs[0]

        encode_code_total = (torch.stack(encode_losses[1:]).sum() if self.n_levels > 1
                              else byte_loss.new_zeros(()))
        encode_total = cfg.byte_ntp_weight * encode_losses[0] + cfg.code_ntp_weight * encode_code_total

        decode_terms = [l for l in decode_losses if l is not None]
        decode_total = (cfg.decode_ntp_weight * torch.stack(decode_terms).sum() if decode_terms
                         else byte_loss.new_zeros(()))

        self_only_terms = [l for l in decode_self_only_losses if l is not None]
        decode_self_only_total = (cfg.decode_self_only_weight * torch.stack(self_only_terms).sum()
                                   if self_only_terms else byte_loss.new_zeros(()))

        loss = encode_total + decode_total + decode_self_only_total
        ntp_total = torch.stack(encode_losses + decode_terms + self_only_terms).sum()
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_acc": byte_acc,
            "encode_total": encode_total, "decode_total": decode_total,
            "decode_self_only_total": decode_self_only_total, "ntp_loss_total": ntp_total,
            **{f"level{i}_ntp_loss_encode": l for i, l in enumerate(encode_losses)},
            **{f"level{i}_ntp_acc_encode": a for i, a in enumerate(encode_accs)},
            **{f"level{i}_ntp_loss_decode": l for i, l in enumerate(decode_losses) if l is not None},
            **{f"level{i}_ntp_acc_decode": a for i, a in enumerate(decode_accs) if a is not None},
            **{f"level{i}_ntp_loss_decode_self": l for i, l in enumerate(decode_self_only_losses) if l is not None},
            **{f"level{i}_ntp_acc_decode_self": a for i, a in enumerate(decode_self_only_accs) if a is not None},
        }

        return loss, metrics


def _sample_next_byte(embed_weight: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
    """embed_weight must come from whichever LM actually produced h_last -- under
    Config.share_level_weights=False, encode and decode no longer share an embedding table, so
    the caller must pass the matching one explicitly (see call sites below)."""
    logits = F.linear(h_last, embed_weight)
    return logits.argmax(-1)


@torch.no_grad()
def generate_no_cache(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str,
                       max_decode_sources: int | None = None) -> torch.Tensor:
    """max_decode_sources: forwarded to RefineLM._run -- None (default) is the full-cumulative
    conditioning mode (this is what the "cond_full" qualitative sample uses). max_decode_sources=1
    forces self-only conditioning (see generate_self_only_cond below).

    FIXED (was: 3 of every 4 generation steps silently fell back to unconditioned encode-only
    prediction for decode_K>1 configs -- see docs/status.md's "generate_no_cache ... ragged-length
    conditioning gap" note for the full original bug writeup). RefineLM._run only activates decode
    for a sequence length divisible by every active track's stride; byte-by-byte generation only
    revisits such a length once every decode_K steps. Fix: pad the growing sequence up to the next
    multiple of decode_K before each _run call (pad VALUE is irrelevant -- see below), then read
    off the REAL last position (index L-1, not the padded tail) for the next-byte prediction.

    Why the padding is safe (not an approximation): causal attention means position L-1's hidden
    state can only depend on positions <= L-1, and the padding is appended strictly AFTER position
    L-1 -- it literally cannot be attended to from there, at any layer, at any level. The one
    subtlety is the FINAL, block that straddles real content and padding (since pad_len < decode_K
    always): that block's OWN code ends up padding-influenced, but that code is only ever used as
    the PREFIX for the NEXT block (strictly after L-1), never for its own bytes -- position L-1's
    decode computation only ever reads PREVIOUS, fully-real blocks' codes (at every level, since
    padding is sized to align every active track's block boundary simultaneously via the full
    product decode_K). So h[:, L-1, :] from the padded call is exactly what a naturally
    block-aligned sequence ending at L would have produced -- not an approximation, mathematically
    identical."""
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    decode_K = 1
    for k in model.cfg.Ks:
        decode_K *= k
    all_bytes = prompt_bytes
    for _ in range(n_new_bytes):
        L = all_bytes.shape[1]
        pad_len = (-L) % decode_K
        padded = (torch.cat([all_bytes, all_bytes.new_zeros(all_bytes.shape[0], pad_len)], dim=1)
                  if pad_len > 0 else all_bytes)
        _, _, _, _, h_list, _, _, _ = model._run(padded, compute_ntp=False, max_decode_sources=max_decode_sources)
        # h_list[0] is decode's output whenever decode ran for level 0 (the normal case once
        # padded -- decode fires at every step), encode's output only in the (rare) fully-ragged
        # fallback -- decode_lms[0].embed is correct for the former; see this function's own
        # padding argument for why decode fires at every step here.
        next_byte = _sample_next_byte(model.decode_lms[0].embed.weight, h_list[0][:, L - 1, :])
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_self_only_cond(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    """Thin wrapper: generate_no_cache with max_decode_sources=1 forced at every step -- level 0's
    decode conditions on ONLY its own code_0 (self track), never level 1's or any coarser level's
    code, regardless of ragged-ness. The mode Config.decode_self_only_aux trains an auxiliary loss
    on; see qualitative_generate's "cond_self" field."""
    return generate_no_cache(model, prompt_bytes, n_new_bytes, device, max_decode_sources=1)


@torch.no_grad()
def generate_encode_only(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    all_bytes = prompt_bytes
    enc0 = model.encode_lms[0]
    for _ in range(n_new_bytes):
        _, _, _, h = enc0(all_bytes, level=0, window=model.windows[0], compute_ntp=False)
        next_byte = _sample_next_byte(enc0.embed.weight, h[:, -1, :])
        all_bytes = torch.cat([all_bytes, next_byte.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return all_bytes[0]


@torch.no_grad()
def generate_level1_codes(model: "RefineLM", prompt_bytes: torch.Tensor, n_new_codes: int, device: str) -> torch.Tensor:
    was_training = model.training
    model.eval()
    prompt_bytes = prompt_bytes.to(device)
    if prompt_bytes.dim() == 1:
        prompt_bytes = prompt_bytes.unsqueeze(0)
    enc0, enc1 = model.encode_lms[0], model.encode_lms[1]
    codes, _, _, _ = enc0(prompt_bytes, level=0, window=model.windows[0], compute_ntp=False)
    n_prompt_codes = codes.shape[1]
    for _ in range(n_new_codes):
        _, _, _, h1 = enc1(codes, level=1, window=model.windows[1], compute_ntp=False)
        logits = F.linear(h1[:, -1, :], enc1.embed.weight)
        next_id = logits.argmax(-1)
        next_code = F.one_hot(next_id, num_classes=model.cfg.vocab).to(codes.dtype)
        codes = torch.cat([codes, next_code.unsqueeze(1)], dim=1)
    if was_training:
        model.train()
    return codes[0, n_prompt_codes:].argmax(-1)


@torch.no_grad()
def level1_ground_truth_codes(model: "RefineLM", full_bytes: torch.Tensor, prompt_len: int, device: str) -> torch.Tensor:
    full_bytes = full_bytes.to(device)
    if full_bytes.dim() == 1:
        full_bytes = full_bytes.unsqueeze(0)
    enc0 = model.encode_lms[0]
    K0 = model.cfg.Ks[0]
    c0, _, _, _ = enc0(full_bytes, level=0, window=model.windows[0], compute_ntp=False)
    ids = c0[0].argmax(-1)
    n_prompt_codes = prompt_len // K0
    return ids[n_prompt_codes:]


def _annotate_bytes_with_codes(byte_ids: torch.Tensor, code_ids: torch.Tensor, K: int) -> str:
    """Group byte_ids into K-byte blocks, tagging each block with its own code_i in {}
    (one code covers K raw bytes, decode_K in RefineLM._run terms)."""
    bytes_list = byte_ids.tolist()
    codes_list = code_ids.tolist()
    parts = []
    for b in range(0, len(bytes_list), K):
        block = "".join(repr(bytes([x]))[2:-1] for x in bytes_list[b:b + K])
        code_idx = b // K
        c = codes_list[code_idx] if code_idx < len(codes_list) else "?"
        parts.append(f"{block}{{{c}}}")
    return "".join(parts)


@torch.no_grad()
def _decode_source_codes(model: "RefineLM", full_bytes: torch.Tensor, device: str, level: int = -1) -> torch.Tensor:
    """Re-derive, per-position, level `level`'s own code_i (see the stream_i/code_i terminology
    note in docs/status.md). level=-1 (default) is the TOPMOST level -- the coarsest of the
    (possibly several, under cumulative multi-level decode) sources level 0's decode conditions
    on; level=0 is level 0's OWN code (the self track, what generate_self_only_cond's output was
    actually conditioned on -- use this, not the topmost, when annotating a self-only sample)."""
    was_training = model.training
    model.eval()
    seq_repr = full_bytes.to(device)
    if seq_repr.dim() == 1:
        seq_repr = seq_repr.unsqueeze(0)
    c_list = []
    for i in range(model.n_levels):
        c_i, _, _, _ = model.encode_lms[i](seq_repr, level=i, window=model.windows[i], compute_ntp=False)
        c_list.append(c_i)
        seq_repr = c_i
    source_c = c_list[level]
    if was_training:
        model.train()
    return source_c.argmax(-1)[0]


def qualitative_generate(model: "RefineLM", prompt_bytes: torch.Tensor, gen_len: int,
                          ground_truth: torch.Tensor | None, device: str, log=print, label: str = "") -> None:
    """Three level0 generation modes, naming scheme mirrors Config.decode_self_only_aux's own
    terminology (see docs/status.md): "uncond" = zero code conditioning (generate_encode_only);
    "cond_self" = self track ONLY, level0's own code_0, forced at every step regardless of
    ragged-ness (generate_self_only_cond -- the mode Config.decode_self_only_aux trains an
    auxiliary loss on, previously untrained/unmeasured entirely); "cond_full" = the full
    cumulative set (self + every coarser level, generate_no_cache -- subject to the
    ragged-length conditioning-gap bug noted in docs/status.md: NOT actually full-conditioned at
    every step, only every step where the running byte count divides every active track's
    stride). Renamed from the earlier "level0_cond"/"level0_cond_codes" (now "cond_full") when
    "cond_self" was added, to keep the three modes unambiguous side by side."""
    prefix = f"qual_{label}_" if label else "qual_"
    out_cond_full = generate_no_cache(model, prompt_bytes, gen_len, device)
    gen_bytes_cond_full = bytes(out_cond_full[prompt_bytes.numel():].tolist())
    out_uncond = generate_encode_only(model, prompt_bytes, gen_len, device)
    gen_bytes_uncond = bytes(out_uncond[prompt_bytes.numel():].tolist())
    log(f"{prefix}prompt:              {bytes(prompt_bytes.tolist())!r}")
    log(f"{prefix}level0_uncond:       {gen_bytes_uncond!r}")
    log(f"{prefix}level0_cond_full:    {gen_bytes_cond_full!r}")
    decode_K = 1
    for k in model.cfg.Ks:
        decode_K *= k
    code_ids_full = _decode_source_codes(model, out_cond_full, device, level=-1)
    n_prompt_codes = prompt_bytes.numel() // decode_K
    gen_code_ids = code_ids_full[n_prompt_codes:]
    annotated = _annotate_bytes_with_codes(out_cond_full[prompt_bytes.numel():], gen_code_ids, decode_K)
    log(f"{prefix}level0_cond_full_codes: {annotated}  <{gen_code_ids.tolist()}>")
    if model.n_levels > 1:
        out_cond_self = generate_self_only_cond(model, prompt_bytes, gen_len, device)
        gen_bytes_cond_self = bytes(out_cond_self[prompt_bytes.numel():].tolist())
        log(f"{prefix}level0_cond_self:    {gen_bytes_cond_self!r}")
        self_K = model.cfg.Ks[0]
        code_ids_self = _decode_source_codes(model, out_cond_self, device, level=0)
        n_prompt_codes_self = prompt_bytes.numel() // self_K
        gen_code_ids_self = code_ids_self[n_prompt_codes_self:]
        annotated_self = _annotate_bytes_with_codes(out_cond_self[prompt_bytes.numel():], gen_code_ids_self, self_K)
        log(f"{prefix}level0_cond_self_codes: {annotated_self}  <{gen_code_ids_self.tolist()}>")
    if ground_truth is not None:
        log(f"{prefix}ground_truth:        {bytes(ground_truth.tolist())!r}")
    if model.n_levels > 1:
        K0 = model.cfg.Ks[0]
        n_new_codes = gen_len // K0
        if n_new_codes > 0:
            level1_gen = generate_level1_codes(model, prompt_bytes, n_new_codes, device)
            log(f"{prefix}level1_gen:          {level1_gen.tolist()}")
            if ground_truth is not None:
                full_bytes = torch.cat([prompt_bytes.reshape(-1), ground_truth.reshape(-1)])
                level1_gt = level1_ground_truth_codes(model, full_bytes, prompt_bytes.numel(), device)
                log(f"{prefix}level1_gt:           {level1_gt.tolist()}")


def load_enwik8(path: Path, n_bytes: int | None = None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read(n_bytes) if n_bytes else f.read()
    return torch.tensor(list(data), dtype=torch.long)


def split_train_val(data: torch.Tensor, val_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_val = max(1, int(len(data) * val_frac))
    return data[:-n_val], data[-n_val:]


def sample_context(data: torch.Tensor, batch_size: int, context_len: int, device: str) -> torch.Tensor:
    n = max(1, len(data) - context_len)
    starts = torch.randint(0, n, (batch_size,))
    return torch.stack([data[s:s + context_len] for s in starts]).to(device)


def lr_at(step: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    return peak


def lr_at_warmup_constant_cosine(
    step: int, warmup: int, constant_steps: int, peak: float, total_steps: int, min_lr_frac: float = 0.1,
) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    decay_start = warmup + constant_steps
    if step < decay_start:
        return peak
    min_lr = peak * min_lr_frac
    progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))


def load_config_module(path: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


@torch.no_grad()
def _add_per_level_bpb(result: dict) -> dict:
    for k in list(result.keys()):
        if k.endswith("_ntp_loss_encode") or k.endswith("_ntp_loss_decode"):
            result[k.replace("_ntp_loss_", "_bpb_")] = result[k] / math.log(2)
    return result


def eval_model(model: RefineLM, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
    model.eval()
    accum: dict[str, list[float]] = {}
    for _ in range(n_batches):
        ctx = sample_context(data, batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.item())
    model.train()
    result = {k: sum(v) / len(v) for k, v in accum.items()}
    result["bpb"] = result["byte_loss"] / math.log(2)
    return _add_per_level_bpb(result)


def build_param_groups(model: RefineLM) -> list[dict]:
    seen: set[int] = set()
    params = []
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    return [{"params": params}]


def train(model: RefineLM, train_data: torch.Tensor, val_data: torch.Tensor, args, log, run_name: str, device: str) -> None:
    opt = torch.optim.AdamW(build_param_groups(model), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.checkpoint_dir / run_name, args.save_every_n_evals, minimize=True)

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train_refine_v4_4", dynamic_ncols=True)
    for step in pbar:
        if args.cosine_decay:
            lr = lr_at_warmup_constant_cosine(step, args.warmup_steps, args.constant_steps, args.lr_peak, args.steps)
        else:
            lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr

        ctx = sample_context(train_data, args.batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        train_bpb = metrics["byte_loss"].item() / math.log(2)
        pbar.set_postfix(
            lr=f"{lr:.2e}", loss=f"{loss.item():.4f}", bpb=f"{train_bpb:.4f}",
            byte_acc=f"{metrics['byte_acc'].item()*100:.2f}%",
        )

        if step % args.log_every == 0:
            train_scalars = {k: v.item() for k, v in metrics.items()}
            train_scalars = _add_per_level_bpb(train_scalars)
            train_scalars["bpb"] = train_bpb
            log(f"{pbar}", step=step, lr=lr, loss=loss.item(),
                **{k: v for k, v in train_scalars.items() if k not in ("loss",)})

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["bpb"])
            log(f"{pbar}  {val_str}  best_val_bpb={checkpointer.best_metric:.4f}",
                step=step, **{f"val_{k}": v for k, v in val.items()}, best_val_bpb=checkpointer.best_metric)

            if args.qual_gen_bytes > 0:
                total_len = args.qual_prompt_bytes + args.qual_gen_bytes
                for label, src_data in (("train", train_data), ("val", val_data)):
                    start = torch.randint(0, max(1, len(src_data) - total_len), (1,)).item()
                    window = src_data[start: start + total_len]
                    qualitative_generate(model, window[: args.qual_prompt_bytes], args.qual_gen_bytes,
                                          window[args.qual_prompt_bytes:], device, log=log, label=label)


def _parse_int_tuple(s) -> tuple[int, ...]:
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    return tuple(int(x) for x in str(s).split(","))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Packed-sequence decode (prepend/interleave), trainable BOS, correct RoPE timing (qcute_refine_v4_4)", parents=[pre])
    p.add_argument("--Ks", default=(32, 32))
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", default=32)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--byte_ntp_weight", type=float, default=1.0)
    p.add_argument("--code_ntp_weight", type=float, default=1.0)
    p.add_argument("--decode_ntp_weight", type=float, default=1.0)
    p.add_argument("--gumbel_tau", type=float, default=1.0)
    p.add_argument("--use_gumbel_noise", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--code_extract_mode", type=str, default="last_h",
                    choices=["last_h", "softmax_pool", "light_query_attn", "query_embed"])
    p.add_argument("--code_head_tied", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--decode_pack_mode", type=str, default="interleave", choices=["interleave", "prepend"])
    p.add_argument("--decode_chunked", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--decode_banded", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--decode_self_only_aux", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--cross_track_source", type=str, default="encode", choices=["encode", "decode"])
    p.add_argument("--decode_self_only_weight", type=float, default=1.0)
    p.add_argument("--decode_code_ste", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--share_level_weights", type=lambda x: x.lower() != "false", default=True)

    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)

    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--cosine_decay", action="store_true")
    p.add_argument("--constant_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--qual_gen_bytes", type=int, default=0)
    p.add_argument("--qual_prompt_bytes", type=int, default=64)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--compile", type=lambda x: x.lower() != "false", default=False)

    # Safety net for the exact bug found this session: decode_code_ste/decode_banded/
    # decode_self_only_aux/decode_self_only_weight were added to Config but never wired into
    # argparse or the Config(...) call below, so config files setting them were SILENTLY ignored
    # (the dataclass default was used instead, no error, no warning) -- e.g. selfcond_detach_k4's
    # entire premise, decode_code_ste=False, never actually took effect. This assert makes that
    # class of bug fail loudly at startup instead of silently invalidating a run's premise: every
    # Config field must have a matching --arg registered above, or be explicitly listed as an
    # intentional CLI-exclusion below.
    _cli_excluded_config_fields: set[str] = set()  # intentionally-excluded field names go here, if any
    _missing_from_cli = ({f.name for f in dataclass_fields(Config)} - {a.dest for a in p._actions}
                          - _cli_excluded_config_fields)
    assert not _missing_from_cli, (
        f"Config field(s) {_missing_from_cli} have no matching --arg registered in main()'s argparse "
        f"setup -- a config .py file setting them would be silently ignored. Add a p.add_argument(...) "
        f"for each, and pass it through to Config(...) below.")

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()
    args.Ks = _parse_int_tuple(args.Ks)

    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(
        Ks=args.Ks, d_model=args.d_model, n_layers=args.n_layers, context_len=args.context_len,
        n_heads=args.n_heads, mlp_mult=args.mlp_mult, attn_window=args.attn_window, rope_base=args.rope_base,
        byte_ntp_weight=args.byte_ntp_weight, code_ntp_weight=args.code_ntp_weight,
        decode_ntp_weight=args.decode_ntp_weight, gumbel_tau=args.gumbel_tau, use_gumbel_noise=args.use_gumbel_noise,
        code_extract_mode=args.code_extract_mode, code_head_tied=args.code_head_tied,
        decode_banded=args.decode_banded, decode_self_only_aux=args.decode_self_only_aux,
        cross_track_source=args.cross_track_source,
        decode_self_only_weight=args.decode_self_only_weight, decode_code_ste=args.decode_code_ste,
        vocab=args.vocab, share_level_weights=args.share_level_weights,
        decode_pack_mode=args.decode_pack_mode, decode_chunked=args.decode_chunked,
    )
    model = RefineLM(cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    if args.compile:
        model = torch.compile(model)

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"qcute_refine_v4_4_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} -- tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} d_model={cfg.d_model} n_layers={cfg.n_layers} seq_lens={model.seq_lens} "
        f"context_len={cfg.context_len} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    log(f"train_bytes={len(train_data)}  val_bytes={len(val_data)}")

    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
