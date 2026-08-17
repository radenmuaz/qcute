"""v5_stack_eff, no-aux 400-step rerun 3/12 of the Ks regression grid (simplest->hardest, see
CLAUDE.md), Ks=(1,1,1), n_layers=1, 1k-byte testbed, val_frac=0.5 (avoids short-val-data
truncation), decode_self_only_aux=False. Exact-match to qcute_v5_stack is not expected (chunked
staircase cross-attention differs from dense at chunk boundaries); note qualitatively how close.

uv run python -m qcute.archive3.qcute_v5_bos --config configs/overfit_stack_eff/qcute_v5_stack_eff_ks111_400_noaux.py
"""
from pathlib import Path

Ks = (1,1,1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (32,32,32)
cross_track_source = "decode"
decode_code_ste = True
share_level_weights = False
use_gumbel_noise = False
gumbel_tau = 1.0

decode_self_only_aux = False

data = Path("datasets/enwik8_1M.gz")
n_bytes = 1000
val_frac = 0.5

steps = 400
batch_size = 16
lr_peak = 6e-4
warmup_steps = 40
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 32
qual_prompt_bytes = 16
