"""qcute.qcute_refine_v4 config: IDENTICAL architecture to configs/
qcute_refine_v4_bpe4_imitate.py (K=4/window=8 at level 0, level 1
window=256 dense + tier_n_layers=(1,2), imitating the 4-layer bytelm/bpelm
baselines' depth) — this file exists only to retrain it under
`qcute_refine_v4.py`'s new (this session) additive PASS1+PASS2 loss
scheme, not to change any architecture/hyperparameter.

Session rationale: `qcute_refine_v4_bpe4_imitate`'s original run (docs/
status.md) was the WORST result of the whole session relative to a
matched baseline — best val_bpb 2.5073, losing to EVERY baseline tried,
including the trivial 1-layer `bytelm_xs1_ctx1024` diagnostic despite 3.2x
the params. That run predates this session's loss-scheme change: PASS 2
used to silently OVERWRITE PASS 1's loss for level 0, so level 0's own
self-attention-only weights (crippled to `attn_window=8`, a near-bag-of-
8-bytes local view) were NEVER directly pushed toward standalone
competence — only level 0's FUSED view ever trained. Given this config's
own extreme level-0 window (the narrowest of any config tried this
session), it's the single most likely candidate to benefit from the new
`fusion_ntp_weight`-controlled additive scheme (`_encode`/`forward` now
sum level0-pass1 + level1-pass1 + level0-pass2 into the loss, all three
differentiable, none discarded — see the module's own docstring and
Config.fusion_ntp_weight) — directly tests the "make it modular, each lm
can act independently and still get good bpb" goal on the config that
most needs it.

Everything else identical to qcute_refine_v4_bpe4_imitate.py: Ks=(4,4),
dqs=(8,8), tier_d_models=(256,256), tier_n_layers=(1,2), context_len=1024,
attn_window=(8,256), byte_repr="embed", code_head_mode="independent",
cross_attn_rope=True, fuse_encoder_levels=True, code_embed_mode="linear"
(default), fusion_ntp_weight=1.0 (default, new this session), steps=4000.

    uv run python -m qcute.qcute_refine_v4 --config configs/qcute_refine_v4_bpe4_imitate_uncond.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_bpe4_imitate_uncond
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 2)
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
