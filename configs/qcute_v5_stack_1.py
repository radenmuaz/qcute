"""v5_stack, Ks=(1,) n_layers=1, full-scale run on the same dataset config as the
qcute.bytelm baseline (bytelm_xs_mtp4_ctx1024.py) except context=256: full enwik8_1M.gz
corpus, val_frac=0.1, steps=8000, batch_size=16, warmup_steps=500, lr_peak=6e-4.
decode_code_ste=True, use_gumbel_noise=False, decode_self_only_aux=False (all Config
defaults, set explicitly here for clarity).

uv run python -m qcute.qcute_v5_stack --config configs/qcute_v5_stack_k1_l1.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_v5_stack_k1_l1
"""
from pathlib import Path

Ks = (2,2,1)
d_model = 256
n_layers = 1
context_len = 1024
attn_window = (16,16,256)
cross_track_source = "decode"
decode_code_ste = True
use_gumbel_noise = False
gumbel_tau = 1.0

decode_self_only_aux = False

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 1000
eval_batches = 20

qual_gen_bytes = 1024
qual_prompt_bytes = 128
