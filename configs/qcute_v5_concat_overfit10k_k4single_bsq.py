"""v5_concat (v4.4.1-derived) starter config, quant_type="bsq" (4-bit code), single-level K=4,
overfit10k testbed. Twin of qcute_v5_concat_overfit10k_k4single.py with softmax -> bsq.

    uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_overfit10k_k4single_bsq.py
"""
from pathlib import Path

Ks = (4,)
d_model = 256
n_layers = 2
context_len = 256
attn_window = (32,)
decode_pack_mode = "interleave"
decode_chunked = True
cross_track_source = "decode"
decode_code_ste = True
share_level_weights = False
use_gumbel_noise = True
gumbel_tau = 2.0
quant_type = "bsq"
bsq_bits = 4

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
qual_prompt_bytes = 64
