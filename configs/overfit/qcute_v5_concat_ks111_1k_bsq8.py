"""v5_concat, bsq quant_type (bsq_bits=8), Ks=(1,1,1), n_layers=1, 1k-byte testbed.
Clone of qcute_v5_concat_ks111_1k.py with quant_type="bsq" instead of the default "softmax" code.

uv run python -m qcute.qcute_v5_concat_slow --config configs/overfit/qcute_v5_concat_ks111_1k_bsq8.py
"""
from pathlib import Path

Ks = (1, 1, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (32, 32, 32)
decode_pack_mode = "interleave"
decode_chunked = True
cross_track_source = "decode"
decode_code_ste = True
share_level_weights = False
use_gumbel_noise = False
gumbel_tau = 1.0

quant_type = "bsq"
bsq_bits = 8

decode_self_only_aux = True

data = Path("datasets/enwik8_1M.gz")
n_bytes = 1000
val_frac = 0.1

steps = 2000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 32
qual_prompt_bytes = 16
