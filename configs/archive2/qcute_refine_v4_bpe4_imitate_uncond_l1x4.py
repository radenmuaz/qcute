"""qcute.qcute_refine_v4 config: CLONE of configs/qcute_refine_v4_bpe4_
imitate_uncond.py, ONE change: `tier_n_layers=(1, 4)` instead of `(1, 2)`
— level 1 gets 4 real self-attention layers, matching `bytelm`/`bpelm`'s
own actual layer count exactly (their baseline configs, e.g. `bytelm_
xs_mtp4_ctx1024`, all use 4 layers), rather than the earlier `bpe4_
imitate*`'s "effective depth ~4 via fusion + 2 layers" approximation.

Session rationale: level 0 stays deliberately SMALL/narrow (`attn_window=
8`, 1 layer) — "pretend level 0 is small" — so essentially all of the
model's real depth lives in level 1 (4 layers, dense/full-context
`window=256`) plus whatever fusion contributes to level 0's own byte
prediction. This is the most literal depth-match to a 4-layer dense
baseline tried this session: not matched depth via a
fusion-plus-shallow-self-attention combination (as `bpe4_imitate`/
`bpe4_imitate_uncond` were), but an ACTUAL 4-layer trunk one level up,
under this session's new additive PASS1+PASS2 loss scheme (`Config.
fusion_ntp_weight`, see qcute_refine_v4.py's own docstring) so level 0's
own standalone (PASS 1) competence is directly trained too, not just an
incidental side effect of fusion.

To report once trained: best val_bpb, mean it/s, params, FLOPs/fwd
(single forward pass, batch=1, `torch.utils.flop_counter.FlopCounterMode`
— same methodology as every other params/FLOPs entry in docs/status.md),
compared against `bytelm_xs_mtp4_ctx1024` (3.412M params, 6.979G FLOPs,
2.3650 best val_bpb) and `qcute_refine_v4_bpe4_imitate`'s own original
result (2.5073, the worst of the session) and `qcute_refine_v4_bpe4_
imitate_uncond`'s new-loss-scheme result (queued ahead of this one).

Everything else identical to qcute_refine_v4_bpe4_imitate_uncond.py:
Ks=(4,4), dqs=(8,8), tier_d_models=(256,256), context_len=1024,
attn_window=(8,256), byte_repr="embed", code_head_mode="independent",
cross_attn_rope=True, fuse_encoder_levels=True, code_embed_mode="linear"
(default), fusion_ntp_weight=1.0 (default), steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_bpe4_imitate_uncond_l1x4.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_bpe4_imitate_uncond_l1x4
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 4)    # the actual ablation vs qcute_refine_v4_bpe4_imitate_uncond.py — level 1: 4 layers
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (8, 256)

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = True
fuse_encoder_levels = True

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 4000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20
