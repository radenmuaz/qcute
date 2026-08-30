"""Static (weight-free) receptive-field connectivity checker for SummFormer -- answers "can raw
input position `source` influence decoder position `target`" as a pure boolean reachability
question over the causal_mask/window conditions, mirroring the real forward pass's masking logic
layer-by-layer but WITHOUT running any tensors/weights.

Why this exists (see chat 2026-08-30): perturb-and-diff empirical testing on random weights
conflates two different questions -- "does a connectivity path exist" (a deterministic fact about
window/stride arithmetic, independent of weights) vs "how strongly is that path weighted after
softmax" (genuinely weight-dependent, and can be so small it looks like zero in float32, or even
float64 at the exact margin). This tool answers only the first question, exactly, with no
randomness, no precision pitfalls -- reachability is transitive-closure-computed via boolean OR
over incoming edges, using the SAME query/key/window predicates windowed_cross_attention/
chunked_windowed_attention/causal_mask enforce (diff < window, causal <=), so it should match a
weight-agnostic idealization of the real forward pass exactly. Confirmed against known empirical
results (self-attn window=2 -> disconnected, window=8-10ish -> connected, cross-attn window=1 with
self-attn window=32 -> connected) before relying on it for anything new -- see __main__ below.

Use empirical perturb-and-diff (check_chain_receptive_field.py) as a SEPARATE follow-up once
connectivity is confirmed here, to gauge whether a real (weighted) path is strong enough to matter
in practice -- this tool only tells you whether it's possible at all.

    uv run python summformer_jax/scripts/check_connectivity.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summformer import StackConfig, ChainStageConfig, CrossAttnSpec, _norm_window, _auto_window


def _self_attn_reachable(reach: np.ndarray, window, n_layers: int) -> np.ndarray:
    """reach[p] = bool, source-reachable at position p. Applies n_layers rounds of
    self-attention edges (p reachable via p' if p'<=p and p-p'<window), matching how stacking
    self-attn layers composes reach round-by-round (mirrors the real n_layers/window compounding
    -- NOT a single-hop shortcut). Vectorized (numpy): a sliding-window OR is a prefix-sum range
    query, O(T) per round instead of the O(T*window) nested-Python-loop original -- same result,
    validated against the original loop version at small scale before trusting at L=150528 (see
    chat 2026-08-30: the original pure-Python nested loops were impractically slow, O(L*n_blocks)
    for the cross-attn part specifically, at real image_classification scale)."""
    T = len(reach)
    w = _auto_window(T) if window in (-1, None) else window
    reach = np.asarray(reach, dtype=bool)
    for _ in range(max(1, n_layers)):
        csum = np.concatenate([[0], np.cumsum(reach.astype(np.int64))])
        idx = np.arange(T)
        lo = np.maximum(0, idx - w + 1)
        window_sum = csum[idx + 1] - csum[lo]
        reach = window_sum > 0
    return reach


def check_connectivity(embed_cfg: StackConfig, chain: tuple, decoder_cfg: StackConfig,
                        cross_specs: tuple, L: int, source: int, target: int) -> dict:
    """Returns {'connected': bool, 'trace': [...]} -- trace lists which stages/layers first
    carried the source position's reachability, for debugging."""
    reach = np.zeros(L, dtype=bool)
    reach[source] = True
    trace = []

    reach = _self_attn_reachable(reach, embed_cfg.window, max(embed_cfg.n_layers, 1) if embed_cfg.n_layers > 0 else 0) \
        if embed_cfg.n_layers > 0 else reach
    trace.append(("embedder", int(reach.sum())))

    stage_reach = []  # per-stage OUTPUT reach arrays, at that stage's own local length
    cum_stride = 1
    cur = reach
    for i, stage in enumerate(chain):
        n_blocks = L // stage.stride if i == 0 else len(cur) // stage.stride
        if n_blocks < 1:
            break
        # point-sample: stage-local position j <- cur[(j+1)*stride - 1]
        idx = (np.arange(n_blocks) + 1) * stage.stride - 1
        sampled = cur[idx]
        sampled = _self_attn_reachable(sampled, stage.window, stage.n_layers)
        cum_stride *= stage.stride
        stage_reach.append((sampled, cum_stride))
        trace.append((f"chain[{i}] (cum_stride={cum_stride})", int(sampled.sum())))
        cur = sampled

    cross_by_layer = {s.dst: s for s in cross_specs}
    dreach = reach.copy()
    q_arr = np.arange(L)
    for i in range(decoder_cfg.n_layers):
        dreach = _self_attn_reachable(dreach, decoder_cfg.window, 1)  # one round per decoder layer
        spec = cross_by_layer.get(i)
        if spec is not None and spec.encoder_output < len(stage_reach):
            code_reach, cs = stage_reach[spec.encoder_output]
            cw = None if spec.force_dense else (_auto_window(cs) if spec.window in (-1, None) else spec.window)
            n_blocks = len(code_reach)
            kpos_arr = (np.arange(n_blocks) + 1) * cs - 1  # sorted increasing
            k_max = np.searchsorted(kpos_arr, q_arr, side="right") - 1  # rightmost kpos <= q
            if cw is None:
                k_min = np.zeros(L, dtype=np.int64)
            else:
                k_min = np.searchsorted(kpos_arr, q_arr - cw + 1, side="left")  # leftmost kpos > q-cw
            pref = np.concatenate([[0], np.cumsum(code_reach.astype(np.int64))])
            hi = np.clip(k_max + 1, 0, n_blocks)
            lo = np.clip(k_min, 0, n_blocks)
            valid_range = k_max >= k_min
            count = np.where(valid_range, pref[hi] - pref[lo], 0)
            dreach = dreach | (count > 0)
        trace.append((f"decoder layer {i}", int(dreach.sum())))

    return {"connected": bool(dreach[target]), "trace": trace}


def check_query_connectivity(embed_cfg: StackConfig, chain: tuple, decoder_cfg: StackConfig,
                              cross_specs: tuple, n_query: int, L: int, source: int) -> dict:
    """QueryClassifierHead-aware variant of check_connectivity: models n_query TRAINABLE query
    positions (placed after the full image, per QueryClassifierHead's real q_pos=arange(L,L+n_query)
    -- so any force_dense/large-window cross-attn spec sees the WHOLE of a stage's output
    unconditionally, not gated by a window relative to some large query position) cross-attending
    into encoder stage outputs, instead of check_connectivity's generic L-length causal decoder
    (which does NOT match QueryClassifierHead's actual topology -- see chat 2026-08-30). Also
    reports each stage's effective code count (n_blocks -- the actual downsample size at that
    depth) and how many stages produced any output at all (n_blocks>=1) for the given L."""
    reach = np.zeros(L, dtype=bool)
    reach[source] = True
    trace = []
    reach = _self_attn_reachable(reach, embed_cfg.window, max(embed_cfg.n_layers, 1)) if embed_cfg.n_layers > 0 else reach
    trace.append(("embedder", int(reach.sum())))

    stage_reach = []  # (reach_array, cum_stride, n_blocks)
    cum_stride = 1
    cur = reach
    for i, stage in enumerate(chain):
        n_blocks = L // stage.stride if i == 0 else len(cur) // stage.stride
        if n_blocks < 1:
            break
        idx = (np.arange(n_blocks) + 1) * stage.stride - 1
        sampled = cur[idx]
        sampled = _self_attn_reachable(sampled, stage.window, stage.n_layers)
        cum_stride *= stage.stride
        stage_reach.append((sampled, cum_stride, n_blocks))
        trace.append((f"chain[{i}] (cum_stride={cum_stride}, n_blocks={n_blocks})", int(sampled.sum())))
        cur = sampled

    cross_by_layer = {s.dst: s for s in cross_specs}
    q_reach = np.zeros(n_query, dtype=bool)  # queries are trainable params -- start unreached
    for i in range(decoder_cfg.n_layers):
        q_reach = _self_attn_reachable(q_reach, decoder_cfg.window, 1)  # trivial among n_query positions
        spec = cross_by_layer.get(i)
        if spec is not None and spec.encoder_output < len(stage_reach):
            code_reach, cs, nb = stage_reach[spec.encoder_output]
            # force_dense (or any window, since q_pos >= L already exceeds every code position by
            # construction) -> unconditionally sees the WHOLE stage output, not window-gated
            q_reach = q_reach | bool(code_reach.any())
        trace.append((f"query-decoder layer {i}", int(q_reach.sum())))

    final_n_blocks = stage_reach[-1][2] if stage_reach else 0
    final_cum_stride = stage_reach[-1][1] if stage_reach else 1
    return {
        "query_reachable": bool(q_reach.any()),
        "trace": trace,
        "n_stages_active": len(stage_reach),
        "final_downsample_size": final_n_blocks,
        "final_cum_stride": final_cum_stride,
        "final_coverage_fraction": final_cum_stride / L,
    }


if __name__ == "__main__":
    L, n_stages, stride = 256, 8, 2

    def build(self_window, cross_window):
        embed_cfg = StackConfig(n_layers=1, window=self_window)
        chain = tuple(ChainStageConfig(stride=stride, n_layers=1, window=self_window) for _ in range(n_stages))
        decoder_cfg = StackConfig(n_layers=n_stages, window=self_window)
        cross = tuple(CrossAttnSpec(dst=i, encoder_output=i, window=cross_window) for i in range(n_stages))
        return embed_cfg, chain, decoder_cfg, cross

    print("=== sanity checks against known empirical results (cross_window=1, matching the actual weighted test) ===")
    for self_w, expect in [(2, False), (3, True), (32, True)]:
        embed_cfg, chain, decoder_cfg, cross = build(self_w, 1)
        r = check_connectivity(embed_cfg, chain, decoder_cfg, cross, L, source=0, target=130)
        status = "OK" if r["connected"] == expect else "MISMATCH"
        print(f"self_window={self_w}, cross_window=1: connected={r['connected']} (expected {expect}) [{status}]")

    embed_cfg, chain, decoder_cfg, cross = build(32, 1)
    r = check_connectivity(embed_cfg, chain, decoder_cfg, cross, L, source=0, target=130)
    print(f"self_window=32, cross_window=1: connected={r['connected']} (expected True)")

    print("\n=== exact minimum self-attn window (cross=-1 auto) ===")
    for w in range(1, 15):
        embed_cfg, chain, decoder_cfg, cross = build(w, -1)
        r = check_connectivity(embed_cfg, chain, decoder_cfg, cross, L, source=0, target=130)
        print(f"self_window={w}: connected={r['connected']}")
