"""qcute_v5_concat_modes_soft: qcute_v5_concat_soft.py cloned onto qcute.qcute_v5_concat,
adding multi_mode_impl="single_pass" on top of the same code_sample_mode="soft" (plain
Gumbel-Softmax relaxation, no hard forward) setup -- otherwise identical (Ks=(1,), context_len=256,
attn_window=(256,)). Note: Ks=(1,) is n_levels=1, so every level has exactly T=1 track (self only)
-- multi_mode_impl is a structural no-op here (no shallower mode exists below T=1,
decode_stage_extra_total stays exactly 0), included for a direct apples-to-apples comparison
against configs/qcute_v5_concat_soft.py's own numbers, not to exercise the multi-mode machinery
itself (see configs/qcute_v5_concat_modes_ks{41,221}.py for that).

uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_modes_soft.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_concat_modes_soft
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (256,)
code_sample_mode = "soft"
multi_mode_impl = "single_pass"
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
