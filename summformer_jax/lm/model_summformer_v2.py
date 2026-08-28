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
  points, each a 5-tuple `(insert_after, stride, window, code_n_layers, source_index)`:
    - insert_after: main_blocks index after which this fuse-stage's cross-attention runs (the
      main stack's own NEXT layer serves as the "refinement" step -- no separate reused block).
    - stride: pooling factor for this stage's own code/summary sequence (same meaning as v1's stride[s]).
    - window: this fuse-stage's cross-attention window (None = unbounded).
    - code_n_layers: depth of this stage's own dedicated "summary LM" (self-attention over the
      pooled sequence) -- own weights, not shared with main_blocks or any other fuse-stage.
    - source_index: where to pool the summary FROM. 0 = the raw embedded stream (before any
      main_blocks layer touches it); -1 = whatever `x` is at this exact insertion point (the
      layer that just ran); a specific int j = main_blocks[j-1]'s output (1-indexed: j=1 is the
      output after main_blocks[0]).
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
    fuse_stages: tuple = ()           # tuple of (insert_after, stride, window, code_n_layers, source_index)
    input_preset: int = 8
    vocab_size: int | None = 50304
    mtp_heads: int = 1
    mtp_weight: float = 1.0
    weight_tie: bool = False
    zero_kv_sink: bool = True
    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32


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
        n_fuse_layers_total = sum(spec[3] * 2 for spec in cfg.fuse_stages)  # code-lm layers + auto-matched fuse layers
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

        self.fuse_stages = nnx.List([
            FuseStageV2(D, cfg.n_heads, cfg.mlp_mult, spec[3], scale_layers, dtype, param_dtype,
                        rngs=rngs, use_sink=cfg.zero_kv_sink)
            for spec in cfg.fuse_stages
        ])
        self.code_lms = nnx.List([
            nnx.List([Block(D, cfg.n_heads, cfg.mlp_mult, scale_layers, dtype, param_dtype,
                             rngs=rngs, use_sink=cfg.zero_kv_sink) for _ in range(spec[3])])
            for spec in cfg.fuse_stages
        ])
        self.code_ln_fs = nnx.List([
            nnx.LayerNorm(D, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs) for _ in cfg.fuse_stages
        ])
        # group fuse-stage indices by their insert_after layer (multiple stages CAN share an
        # insertion point -- applied in the order they appear in cfg.fuse_stages)
        self.insertions: dict[int, list[int]] = {}
        for i, spec in enumerate(cfg.fuse_stages):
            self.insertions.setdefault(spec[0], []).append(i)

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
        """Returns the code-LM's output after EACH layer (not just the final one) -- FuseStageV2
        cross-attends layer l to this list's element l, so fuse depth auto-matches code depth."""
        outs = []
        for block in self.code_lms[stage_i]:
            code_h = block(code_h, cos, sin, mask, self.cfg.pos_method)
            outs.append(self.code_ln_fs[stage_i](code_h))
        return outs

    def _pool_and_fuse(self, stage_i: int, x, x0, layer_hist, seq_pos, cos_b, sin_b):
        cfg = self.cfg
        hd = self.head_dim
        insert_after, stride, window, code_n_layers, source_index = cfg.fuse_stages[stage_i]
        source = x0 if source_index == 0 else (x if source_index == -1 else layer_hist[source_index])

        L = source.shape[1]
        n_blocks = L // stride
        if n_blocks < 1:
            return x
        code_h = source[:, stride - 1::stride, :][:, :n_blocks, :]
        code_local_pos = jnp.arange(n_blocks)
        cos_c, sin_c = (rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base) if cfg.pos_method == "rope" else (None, None))
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
