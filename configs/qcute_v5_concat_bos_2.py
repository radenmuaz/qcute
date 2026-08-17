"""qcute_v5_concat_2 baseline: plain softmax/argmax quantization, Ks=(4,1), context_len=256,
attn_window=(16,64) -- concat-decode (qcute.qcute_v5_concat) counterpart of configs/qcute_v5_2.py,
for a same-hyperparameters comparison against the staged cross-attention decode (qcute.qcute_v5).
Comparison point for qcute_v5_concat_2_gumbel.py (gumbel noise),
qcute_v5_concat_2_bsq8.py/qcute_v5_concat_2_bsq16.py (BSQ quantization).

uv run python -m qcute.archive3.qcute_v5_concat_bos --config configs/qcute_v5_concat_bos_2.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_concat_bos_2
"""
from pathlib import Path

Ks = (4,1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (16,64)
use_gumbel_noise = False
# gumbel_tau = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 2000
# eval_every = 100
eval_batches = 20

qual_gen_bytes = 128
qual_prompt_bytes = 64
