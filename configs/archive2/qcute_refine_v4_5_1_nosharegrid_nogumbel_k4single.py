"""qcute_refine_v4_5_1 twin of qcute_refine_v4_5_nosharegrid_nogumbel_k4single.py, using LM-
continuation self-code decode instead of v4.5's staged cross-attention. See
qcute/qcute_refine_v4_5_1.py's module docstring and docs/status.md's "Decode redesign: self-code
LM continuation" section for the full design rationale.

    uv run python -m qcute.qcute_refine_v4_5_1 --config configs/qcute_refine_v4_5_1_nosharegrid_nogumbel_k4single.py

    # watch live:
    tail -f logs/qcute_refine_v4_5_1_nosharegrid_nogumbel_k4single/run.log
"""
from pathlib import Path

Ks = (4,)
d_model = 256
n_layers = 2
context_len = 256
attn_window = ((8, 256),)
cross_track_source = "decode"
decode_code_ste = False
share_level_weights = False
use_gumbel_noise = False
gumbel_tau = 1.0

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
