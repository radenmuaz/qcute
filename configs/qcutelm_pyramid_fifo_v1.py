"""qcute.qcutelm_pyramid config: fifo_v1 mode — a faithful port of the now-
deleted qcute_fifo.py's actual v1 mechanism (session: "merge the current
qcute_fifo greedy merge algorithm as flag then delete"), running inside
qcutelm_pyramid.py instead of its own module. See qcutelm_pyramid.py's
Config.mode docstring for the full port (composition-sampled window,
unquantized recursive linear merge, Fetch-style byte-chain MTP head —
all ported verbatim from the retired file).

Renamed from configs/qcute_fifo_w32_bw2.py — its own history is preserved
below (previously revised from window=32/bandwidths=(1,2)'s ~32-64 byte
achievable span, far short of the other baselines' fixed 1024-byte
context):

  rank  window  bandwidths      span_range     avg_span  attn_cost(window^2)  notes
    1    256    (1,2,4)         256-1024        597 (58%)     65536           max=1024 exact, 3 levels, 4x cheaper than rank 2   <- CHOSEN
    2    512    (1,2)           512-1024        768 (75%)    262144           tightest bracket, but only 2 levels, 4x attn cost
    3    128    (1,2,4,8)       128-1024        480 (47%)     16384           cheap, but min_span only 12.5% of 1024
    4     64    (1,2,4,8,16)     64-1024        397 (39%)      4096           cheapest, but min_span only 6.25% of 1024
    5    768    (1,2)          768-1536       1152 (112%)    589824           overshoots past 1024, most expensive

Chosen: window=256, bandwidths=(1,2,4) — caps at exactly 1024 like the
tightest option, at 1/4 the attention cost, exercising the full 3-level
compression design.

    uv run python -m qcute.qcutelm_pyramid --config configs/qcutelm_pyramid_fifo_v1.py
"""
from pathlib import Path

mode = "fifo_v1"
window = 256
bandwidths = (1, 2, 4)

d_model = 256
n_heads = 4
n_layers = 4
mlp_mult = 4
fetch_n_heads = 4
fetch_gamma = 1.0
tie_head = True

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False
constant_steps = 100
eval_every = 100
eval_batches = 20
