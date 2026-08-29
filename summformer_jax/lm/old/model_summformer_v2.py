"""v2: GPT-style deep backbone with FuseStage cross-attention spliced in at chosen depths, NO
weight sharing anywhere (contrast with model_summformer.py/model_summformer_v1.py's small
n_layers + uniform stride-cascade with a reused block for every refinement pass).

Semantics differences from v1:
- `n_layers` main blocks are SEQUENTIAL and DISTINCT (own weights each), like a plain causal
  transformer stack (e.g. n_layers=12 ~ gpt2-small depth) -- this is `main_blocks`, referred to
  below as "main self-attention" to distinguish it from the fuse-stage machinery.
- v1's periodic cascade (run all n_fuse stages every forward pass, always reusing the SAME
  n_layers-sized block for both the initial pass and every post-cross-attn refinement) is replaced
  by explicit placement: `Config.fuse_stages` is a tuple of independently-configured insertion
  points, each `((src, dst), (stride, window), (code_n_layers, code_d_model=None, code_n_heads=None))`
  -- matches image_gen summformer's fuse_stages format, see `_parse_fuse_stage`:
    - dst (insert_after): main_blocks index after which this fuse-stage's cross-attention runs (the
      main stack's own NEXT layer serves as the "refinement" step -- no separate reused block).
    - src (source_index): where to pool the summary FROM. 0 = the raw embedded stream (before any
      main_blocks layer touches it); -1 = whatever `x` is at this exact insertion point (the
      layer that just ran); a specific int j = main_blocks[j-1]'s output (1-indexed: j=1 is the
      output after main_blocks[0]).
    - stride: pooling factor for this stage's own code/summary sequence (same meaning as v1's stride[s]).
    - window: this fuse-stage's cross-attention window (None = unbounded).
    - code_n_layers: depth of this stage's own dedicated "summary LM" (self-attention over the
      pooled sequence) -- own weights, not shared with main_blocks or any other fuse-stage.
    - code_d_model/code_n_heads: optional per-stage code-LM width/head-count override, omitted ->
      defaults to trunk dims; when given, code_in_proj/code_out_proj bridge trunk dim <-> code dim
      at the fuse-stage boundary (FuseStageV2's own cross-attention stays trunk-dim only).
  The FuseStage's own cross-attention depth is auto-matched to code_n_layers, not independently
  configurable -- FuseStageV2's layer l cross-attends to the code-LM's OWN intermediate output
  after ITS layer l (l=0..code_n_layers-1), not a single frozen final summary reused at every fuse
  layer. At code_n_layers=1 this degenerates to the original single-cross-attention behavior.
  Every fuse-stage gets its own FuseStageV2 module and its own code-LM Block stack -- nothing here
  is shared, tied, or reused, by design.
- Since FLOPs don't depend on whether weights are shared (a forward pass through a block costs the
  same regardless), matching v1's FLOPs at the same nominal structure is automatic; matching v1's
  PARAM count is not -- v2 configs deliberately use a SHALLOWER main_layers than the literal
  baseline depth (see configs/summformer_jax_v2/*.py) specifically to compensate for the extra
  params that come from no longer reusing one block 4x for the price of one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from model_summformer import (
    ROPE_PRESETS, Attn, MLP, Block, causal_mask,
    cross_entropy, rope_cos_sin_for_positions,
)


class FuseStageV2(nnx.Module):
    """Like model_summformer.py's FuseStage, but layer l cross-attends to the code-LM's OWN
    intermediate output after its layer l, not one frozen summary reused at every layer -- depth
    is always auto-matched to code_n_layers (see ConfigV2.fuse_stages docstring above)."""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, n_layers: int, scale_layers: int,
                 dtype, param_dtype, *, rngs: nnx.Rngs, use_sink: bool = True):
        self.ln1 = nnx.List([nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs) for _ in range(n_layers)])
        self.attn = nnx.List([Attn(d_model, n_heads, scale_layers, dtype, param_dtype, rngs=rngs, use_sink=use_sink) for _ in range(n_layers)])
        self.ln2 = nnx.List([nnx.LayerNorm(d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs) for _ in range(n_layers)])
        self.mlp = nnx.List([MLP(d_model, mlp_mult, scale_layers, dtype, param_dtype, rngs=rngs) for _ in range(n_layers)])

    def __call__(self, x, code_kv_list, cos_q, sin_q, cos_k, sin_k, attn_mask, pos_method: str) -> jnp.ndarray:
        assert len(code_kv_list) == len(self.attn)
        for l in range(len(self.attn)):
            xn = self.ln1[l](x)
            coden = self.ln1[l](code_kv_list[l])
            x = x + self.attn[l].forward_cross(xn, coden, cos_q, sin_q, cos_k, sin_k, attn_mask, pos_method)
            x = x + self.mlp[l](self.ln2[l](x))
        return x


@dataclass
class ConfigV2:
    n_layers: int = 12                # main backbone depth, distinct weights each layer
    d_model: int = 768
    n_heads: int = 12
    mlp_mult: int = 4
    pos_method: str = "rope"
    rope_base: float = 10000.0
    rope_preset: str | None = None
    context_len: int = 1024
    main_window: int | tuple | None = None   # int (all layers), tuple of len n_layers, or None
    # Each stage: ((src, dst), (stride, window), (code_n_layers, code_d_model=None, code_n_heads=None))
    #   src = source_index (0 = raw embedded x0; -1 = current x at this insertion point; specific
    #         int j = main_blocks[j-1]'s output). dst = insert_after (main_blocks index this stage's
    #         cross-attention fires after). code_d_model/code_n_heads omitted -> default to trunk
    #         dims; given -> the code-LM runs at its own (typically wider) dim, bridged to/from
    #         trunk dim at the fuse-stage boundary via code_in_proj/code_out_proj (Attn/FuseStageV2
    #         themselves stay trunk-dim only). See _parse_fuse_stage. Matches image_gen semantics.
    fuse_stages: tuple = ()
    input_preset: int = 8
    vocab_size: int | None = 50304
    mtp_heads: int = 1
    mtp_weight: float = 1.0
    weight_tie: bool = False
    zero_kv_sink: bool = True
    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32


def _parse_fuse_stage(spec):
    """spec: ((src, dst), (stride, window), (code_n_layers, code_d_model=None, code_n_heads=None)).
    window: -1 = unbounded (converted to None for causal_mask). code_d_model/code_n_heads omitted
    -> None, caller defaults to trunk dims. code_n_layers is always required (first in its group,
    not last, so the optional width/head-count pair trails a variable-length tuple instead of
    sitting in the middle)."""
    (src, dst), (stride, window), code_part = spec
    window = None if window == -1 else window
    code_n_layers = code_part[0]
    code_d_model = code_part[1] if len(code_part) > 1 else None
    code_n_heads = code_part[2] if len(code_part) > 2 else None
    return src, dst, stride, window, code_n_layers, code_d_model, code_n_heads


def _resolve_main_windows(w, n_layers: int) -> tuple:
    if isinstance(w, (tuple, list)):
        assert len(w) == n_layers, f"main_window tuple must have length n_layers={n_layers}, got {len(w)}"
        return tuple(w)
    return (w,) * n_layers


class SummTransformerV2(nnx.Module):
    def __init__(self, cfg: ConfigV2, *, rngs: nnx.Rngs):
        if cfg.rope_preset is not None:
            cfg.rope_base = ROPE_PRESETS[cfg.rope_preset]
        assert cfg.pos_method in ("rope", "learnable", "base")
        self.cfg = cfg
        D = cfg.d_model
        self.head_dim = D // cfg.n_heads
        V = cfg.vocab_size if cfg.vocab_size is not None else 2 ** cfg.input_preset
        self.vocab = V
        assert D % cfg.n_heads == 0
        dtype, param_dtype = cfg.compute_dtype, cfg.param_dtype

        # NANOGPT_SCALE_INIT: every main layer AND every fuse-stage/code-lm layer is a genuine,
        # independent residual-stream-additive block here (no reuse), so scale_layers is simply
        # the total count of all of them -- the true "effective depth" of this architecture.
        n_fuse_layers_total = sum(_parse_fuse_stage(spec)[4] * 2 for spec in cfg.fuse_stages)  # code-lm layers + auto-matched fuse layers
        scale_layers = cfg.n_layers + n_fuse_layers_total

        init = nnx.initializers.normal(stddev=0.02)
        self.embed = nnx.Embed(V, D, embedding_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.wpe = (
            nnx.Embed(cfg.context_len, D, embedding_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            if cfg.pos_method == "learnable" else None
        )

        self.main_blocks = nnx.List(
            [Block(D, cfg.n_heads, cfg.mlp_mult, scale_layers, dtype, param_dtype, rngs=rngs, use_sink=cfg.zero_kv_sink)
             for _ in range(cfg.n_layers)])
        self.main_windows = _resolve_main_windows(cfg.main_window, cfg.n_layers)
        self.ln_f = nnx.LayerNorm(D, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)

        # per-stage code-LM dim: defaults to trunk dim D when code_d_model/code_n_heads are omitted
        # from a stage's spec, or its own (larger, typically) dim when given -- code-LM runs
        # cheaply-infrequently (once per stride trunk positions) so it can afford to be
        # wider/deeper than the trunk without dominating cost. Attn/FuseStageV2 themselves stay
        # trunk-dim only; code_in_proj/code_out_proj bridge the boundary so no cross-attention
        # machinery needs to support mixed dims directly.
        parsed = [_parse_fuse_stage(spec) for spec in cfg.fuse_stages]
        code_dims = [(cd if cd is not None else D) for _, _, _, _, _, cd, _ in parsed]
        code_n_heads_list = [(ch if ch is not None else cfg.n_heads) for _, _, _, _, _, _, ch in parsed]
        for cd, ch in zip(code_dims, code_n_heads_list):
            assert cd % ch == 0, f"code_d_model={cd} must be divisible by code_n_heads={ch}"
        self.code_dims = code_dims
        self.code_head_dims = [cd // ch for cd, ch in zip(code_dims, code_n_heads_list)]

        self.fuse_stages = nnx.List([
            FuseStageV2(D, cfg.n_heads, cfg.mlp_mult, p[4], scale_layers, dtype, param_dtype,
                        rngs=rngs, use_sink=cfg.zero_kv_sink)
            for p in parsed
        ])
        self.code_lms = nnx.List([
            nnx.List([Block(cd, ch, cfg.mlp_mult, scale_layers, dtype, param_dtype,
                             rngs=rngs, use_sink=cfg.zero_kv_sink) for _ in range(p[4])])
            for p, cd, ch in zip(parsed, code_dims, code_n_heads_list)
        ])
        self.code_ln_fs = nnx.List([
            nnx.LayerNorm(cd, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs) for cd in code_dims
        ])
        # projections at the trunk<->code-LM boundary -- identity-shaped (D->D) when code_d_model==D,
        # still allocated for a uniform code path (cheap, one Linear per stage either way).
        self.code_in_proj = nnx.List([
            nnx.Linear(D, cd, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            for cd in code_dims
        ])
        self.code_out_proj = nnx.List([
            nnx.Linear(cd, D, use_bias=True, kernel_init=init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            for cd in code_dims
        ])
        # group fuse-stage indices by their insert_after (dst) layer (multiple stages CAN share an
        # insertion point -- applied in the order they appear in cfg.fuse_stages)
        self.insertions: dict[int, list[int]] = {}
        for i, p in enumerate(parsed):
            self.insertions.setdefault(p[1], []).append(i)  # p[1] = dst (insert_after)

        self.head = (
            nnx.Linear(D, V, use_bias=False, kernel_init=init, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
            if not cfg.weight_tie else None
        )
        self.weight_tie = cfg.weight_tie
        self.extra_heads = nnx.List(
            [nnx.Linear(D, V, use_bias=False, kernel_init=init, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
             for _ in range(max(0, cfg.mtp_heads - 1))])

    def _head_weight(self) -> jnp.ndarray:
        return self.embed.embedding.value if self.weight_tie else self.head.kernel.value.T  # type: ignore[union-attr]

    def _run_code_lm(self, stage_i: int, code_h, cos, sin, mask) -> list:
        """code_h/outs are at this stage's OWN code_d_model (post code_in_proj); returned list is
        projected back to trunk dim (code_out_proj) so FuseStageV2's cross-attention -- trunk-dim
        only -- needs no changes for mixed dims."""
        outs = []
        for block in self.code_lms[stage_i]:
            code_h = block(code_h, cos, sin, mask, self.cfg.pos_method)
            normed = self.code_ln_fs[stage_i](code_h)
            outs.append(self.code_out_proj[stage_i](normed))
        return outs

    def _pool_and_fuse(self, stage_i: int, x, x0, layer_hist, seq_pos, cos_b, sin_b):
        cfg = self.cfg
        hd = self.head_dim
        code_hd = self.code_head_dims[stage_i]
        source_index, _insert_after, stride, window, _code_n_layers, _, _ = _parse_fuse_stage(cfg.fuse_stages[stage_i])
        source = x0 if source_index == 0 else (x if source_index == -1 else layer_hist[source_index])

        L = source.shape[1]
        n_blocks = L // stride
        if n_blocks < 1:
            return x
        code_h = self.code_in_proj[stage_i](source[:, stride - 1::stride, :][:, :n_blocks, :])
        code_local_pos = jnp.arange(n_blocks)
        cos_c, sin_c = (rope_cos_sin_for_positions(code_local_pos, code_hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
        code_mask = causal_mask(code_local_pos, code_local_pos, None)
        h_code_list = self._run_code_lm(stage_i, code_h, cos_c, sin_c, code_mask)

        code_pos_abs = (jnp.arange(n_blocks) + 1) * stride - 1
        cos_k, sin_k = (rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
        fuse_mask = causal_mask(seq_pos, code_pos_abs, window)
        return self.fuse_stages[stage_i](x, h_code_list, cos_b, sin_b, cos_k, sin_k, fuse_mask, cfg.pos_method)

    def _cascade(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        B, L = token_ids.shape
        hd = self.head_dim
        pm = cfg.pos_method

        seq_pos = jnp.arange(L)
        cos_b, sin_b = (rope_cos_sin_for_positions(seq_pos, hd, cfg.rope_base) if pm == "rope" else (None, None))

        x0 = self.embed(token_ids)
        if pm == "learnable":
            x0 = x0 + self.wpe(seq_pos)[None]

        x = x0
        layer_hist = [x0]  # layer_hist[j] = output after main_blocks[j-1] for j>=1; [0] = x0
        for stage_i in self.insertions.get(0, []):
            x = self._pool_and_fuse(stage_i, x, x0, layer_hist, seq_pos, cos_b, sin_b)

        for i, block in enumerate(self.main_blocks):
            seq_mask = causal_mask(seq_pos, seq_pos, self.main_windows[i])
            x = block(x, cos_b, sin_b, seq_mask, pm)
            layer_hist.append(x)
            for stage_i in self.insertions.get(i + 1, []):
                x = self._pool_and_fuse(stage_i, x, x0, layer_hist, seq_pos, cos_b, sin_b)

        return self.ln_f(x)

    def __call__(self, token_ids: jnp.ndarray) -> tuple:
        cfg = self.cfg
        L = token_ids.shape[1]
        x = self._cascade(token_ids)

        w = self._head_weight()
        logits = x[:, :-1, :] @ w.T
        targets = token_ids[:, 1:]
        loss = cross_entropy(logits, targets)

        mtp_losses, mtp_accs = [], []
        for i, head_i in enumerate(self.extra_heads):
            k = i + 2
            if L <= k:
                continue
            logits_i = head_i(x[:, :-k, :])
            targets_i = token_ids[:, k:]
            mtp_losses.append(cross_entropy(logits_i, targets_i))
            mtp_accs.append((jnp.argmax(logits_i, axis=-1) == targets_i).astype(jnp.float32).mean())

        total_loss = loss
        if mtp_losses:
            total_loss = total_loss + cfg.mtp_weight * jnp.mean(jnp.stack(mtp_losses))

        metrics = {
            "loss": total_loss, "final_loss": loss, "bpb": loss / math.log(2),
            **{f"mtp{i+2}_loss": l for i, l in enumerate(mtp_losses)},
            **{f"mtp{i+2}_acc": a for i, a in enumerate(mtp_accs)},
        }
        return total_loss, metrics
