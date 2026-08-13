"""qcute_refine_v4_5_1_nosharegrid_k1.py twin with decode_ntp_weight=0.0 -- isolates whether
encode_lms's uncond generation quality is being dragged down by decode's gradient (via any shared
computation), same diagnostic as the earlier v4.5/v4.4 uncondonly sanity checks (docs/status.md).

    uv run python -m qcute.qcute_refine_v4_5_1 --config configs/qcute_refine_v4_5_1_uncondonly_k1.py
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 2
context_len = 256
attn_window = (32,)
cross_track_source = "decode"
decode_code_ste = False
share_level_weights = False
use_gumbel_noise = True
gumbel_tau = 2.0
decode_ntp_weight = 0.0

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
