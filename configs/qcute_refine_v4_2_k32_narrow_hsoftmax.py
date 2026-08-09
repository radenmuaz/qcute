"""qcute.qcute_refine_v4_2 config: full-width (`bit_inner_downsample=1`)
`bit_head_class="hsoftmax"` — first training run of `BitPredictHeadHSoftmax`
(new this session, from "find something that satisfy chain probs
validity and cheap and same repr power as large softmax head"; downsample
support added/verified per follow-up "implement bitpredicthsoftmax, but
allow downsample").

Classic hierarchical softmax (Morin & Bengio 2005) over the same
`dq`-depth binary tree every BitPredictHead* factorizes — unlike attn/
conv/ssm (one classifying direction per bit POSITION, shared across
every prefix reaching that position — the diagnosed bottleneck), gives
every one of the `2**dq-1` tree NODES its own private weight vector: the
same order of degrees of freedom as a dense `softmax-256` (`65,535` vs
`65,536` params) while only touching the `dq=8` nodes on each example's
own true path — `4,224` FLOPs/example vs. dense softmax's `131,072`
(31x fewer, measured via FlopCounterMode).

`bit_inner_downsample` deliberately left at its default (1, full width)
— session follow-up measurement found downsampling this specific head is
COUNTERPRODUCTIVE on both FLOPs and wallclock (unlike every other chain
head): the added `in_proj` (`D->d_inner`) costs more than hsoftmax's own
tiny native node-read, so `downsample=4` costs MORE FLOPs (33,920 vs
4,224) and is SLOWER (0.063ms/fwd vs 0.049ms) than full width, for half
the params. Full width wins on every axis except raw param count here —
verified via FlopCounterMode + wallclock benchmark, downsample support
itself verified separately (fixed/loop consistency, full-model
integration, `validate_generation` parity at `downsample=4`) in case a
future run wants the param tradeoff specifically.

`code_embed_mode="pq_table"` carried over — same fix that resolved
`attn_id4`'s original divergence.

Everything else identical to this session's other `k32_narrow` chain-head
configs: Ks=(32,32), dq=8, d_model=256, n_layers=1, context_len=1024,
attn_window=(32,32), fuse_encoder_levels=True, fuse_use_null_kv=False,
code_head_mode="chain", steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_hsoftmax.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_hsoftmax
"""
from pathlib import Path

Ks = (32, 32)
dq = 8
d_model = 256
n_layers = 1
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

fuse_encoder_levels = True
fuse_use_null_kv = False
code_head_mode = "chain"
bit_head_class = "hsoftmax"
bit_inner_downsample = 1
code_embed_mode = "pq_table"

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
