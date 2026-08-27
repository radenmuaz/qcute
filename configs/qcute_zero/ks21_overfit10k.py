"""qcute_zero/ks21_overfit10k: first qcute_zero run -- Ks=(2,1), one fuse stage (period 2 bytes).
Single shared LM (byte pass + this stage's own code-sequence NTP pass reuse the exact same
blocks/embed), zero-KV sink mandatory on every attention call, NO curriculum (see
qcute_zero.py's module docstring for why none should be needed by design -- every fuse stage's
code source is the same already-training backbone from step 1, and the sink lets under-trained
fuse-attention weights self-suppress early on). Matches qcute_lagcodec's ks21_v256_pq1_overfit10k.py in
scale/methodology (n_bytes=10000, context_len=256) for direct comparison.

uv run python -m qcute.qcute_zero.qcute_zero --config configs/qcute_zero/ks21_overfit10k.py
"""
from pathlib import Path

run_name = "qcute_zero_ks21_overfit10k"
Ks = (2, 1)
d_model = 256
n_layers = 2
n_heads = 4
context_len = 256
attn_window = None
fuse_window = None
input_preset = 8

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
log_every = 20
eval_every = 50
eval_batches = 5
