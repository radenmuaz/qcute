"""qcute.qcute_refine_v2 config: single byte-LM + single code-LM tower
("only single code lm exist, i.e. byte, code"), sized for a fair
capacity/receptive-field comparison against bytelm_xs_mtp4_ctx1024.

Ks=(4,4): byte LM (level 0) emits its own code every 4 bytes.
context_len=1024, so the code LM's (level 1's) own sequence length is
1024/4 = 256 — "code lm ctx 256 as the last encoder... 4*256=1024
effective ctx len." Ks[1]=4 is otherwise a don't-care (level 1 is the
top/last encoder — nothing consumes its own emitted code further).

attn_window=(256, 64): PER-LEVEL now (qcute_refine_v2.py's Config.attn_window
accepts either a single broadcast int or a per-level tuple) — level 0
(byte LM, seq=1024) windows at 256 (genuine chunked path: 1024%256==0,
1024>256), level 1 (code LM, seq=256) windows at 64 (also genuine
chunked: 256%64==0, 256>64). Session revision: an earlier attempt shared
ONE scalar (256) across both levels, which gave level 1 a window exactly
equal to its own sequence length — mathematically indistinguishable from
full dense (T>window false -> fallback, and a 1-chunk "windowed" pass
computes the identical thing anyway), not a meaningful choice. Per-level
windows let the top level genuinely window too, as asked ("allow
exception to highest layer to force using window attention") — level 1
now has real, non-trivial local reach (2*64=128 own positions =
128*4=512 raw-byte-equivalent) rather than automatic full coverage.

tier_d_models=(256,256): "make lm dim large like bytelm as fair
compare" — matches bytelm_xs_mtp4_ctx1024's own d_model=256 exactly
(n_heads=4, head_dim=64, same as bytelm's xs preset). tok_d_model=256 to
match. tier_n_layers stays (1,1) (width parity, not depth parity).
Resulting params: 3.04M (vs. bytelm xs's ~3.7M) — same order of magnitude.

BUG FOUND AND FIXED this session, load-bearing for this config
specifically: at d_model=256, `nn.MultiheadAttention`'s MPS backward pass
produced NaN gradients (confirmed via named_parameters() localization:
`encoders.0.ntp_head.self_attn.out_proj.weight.grad`; confirmed MPS-
specific — an identical run was perfectly stable on CPU). Every other
attention op in this file already used manual
`F.scaled_dot_product_attention`; BitPredictHead and CrossBlock's
internal `nn.MultiheadAttention` usage has been rewritten the same way,
which resolved it (50-step MPS smoke test, no NaN). This didn't surface
at the earlier d_model=96 configs' scale — a real, size-dependent MPS
kernel bug, not an architecture instability.

Optimizer/LR/steps: same convention as every other config here — matched
to bytelm_xs_mtp4_ctx1024 (steps=8000, batch_size=16, lr_peak=6e-4,
warmup_steps=500, cosine_decay=False).

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_v2_byte4_code256.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v2_byte4_code256
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (256, 64)

tok_d_model = 256
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20
