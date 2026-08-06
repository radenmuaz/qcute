"""qcute.qcute_fifo config — FILENAME NOW STALE (kept as qcute_fifo_w32_bw2.py
only because run_queue3.sh's orchestrator already hardcodes this path and
bytelm_xs_mtp4_ctx1024 was 87% done when this was revised — rename to
qcute_fifo_w256_bw4.py next time the queue script itself gets rebuilt).

Revised from the original window=32/bandwidths=(1,2) first trial: that
config's achievable byte span was only 32-64 bytes (window * bandwidth),
16-32x short of the other baselines' fixed 1024-byte context — not
remotely comparable. Picked from this ranked set of (window, bandwidths)
combinations, ranked by how tightly [min_span, max_span] brackets 1024
(max_span == 1024 first, then avg_span closeness, then attention cost —
computed via enumerate_compositions(window, bandwidths), see qcute_fifo.py):

  rank  window  bandwidths      span_range     avg_span  attn_cost(window^2)  notes
    1    256    (1,2,4)         256-1024        597 (58%)     65536           max=1024 exact, 3 levels, 4x cheaper than rank 2   <- CHOSEN
    2    512    (1,2)           512-1024        768 (75%)    262144           tightest bracket, but only 2 levels, 4x attn cost
    3    128    (1,2,4,8)       128-1024        480 (47%)     16384           cheap, but min_span only 12.5% of 1024
    4     64    (1,2,4,8,16)     64-1024        397 (39%)      4096           cheapest, but min_span only 6.25% of 1024 — back
                                                                                toward the original mismatch problem
    5    768    (1,2)          768-1536       1152 (112%)    589824           overshoots past 1024 instead of capping at it;
                                                                                most expensive

Chosen: window=256, bandwidths=(1,2,4) (rank 1) — caps at exactly 1024
like the tightest option, at 1/4 the attention cost, and exercises the
full 3-level compression design from the original worked example rather
than truncating to 2 levels. bandwidths=(1,2,4) was already sanity-tested
earlier this session (forward/backward, clean gradients).

    uv run python -m qcute.qcute_fifo --config configs/qcute_fifo_w32_bw2.py
"""
from pathlib import Path

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
