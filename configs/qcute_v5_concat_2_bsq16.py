"""qcute_v5_concat_2 variant: BSQ quantization (bsq_bits=16, the CodeEmbed table-lookup ceiling --
MAX_PQ_TABLE_DQ in qcute/qcute_v5_concat.py) instead of the softmax/gumbel categorical code,
otherwise identical to configs/qcute_v5_concat_2.py (Ks=(4,1), context_len=256,
attn_window=(16,64)) -- concat-decode (qcute.qcute_v5_concat) counterpart of
configs/qcute_v5_2_bsq16.py.

uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_2_bsq16.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_concat_2_bsq16
"""
from pathlib import Path

Ks = (4,1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (16,64)
use_gumbel_noise = False
quant_type = "bsq"
bsq_bits = 16
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
