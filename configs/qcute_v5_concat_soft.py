"""qcute_v5_concat_soft: plain Gumbel-Softmax relaxation (code_sample_mode="soft" -- Jang et al.
2016: gumbel-noise softmax used directly in the forward pass, no hard one-hot/straight-through)
variant of configs/qcute_v5_concat_3.py, otherwise identical (Ks=(1,), context_len=256,
attn_window=(256,)). Sibling of configs/qcute_v5_concat_gumbel.py (code_sample_mode="sample" --
same noise, but hard-forward/straight-through instead of soft-forward).
Note: check_gen_consistency is EXPECTED to show many mismatches under gumbel noise -- the
teacher-forced pass draws one batched gumbel-noise tensor over the whole sequence, while the
no-cache incremental path recomputes from scratch (fresh noise draw) at every step, so the two
code paths' RNG streams structurally diverge; not a bug (see docs/status.md's qcute_v5_2_gumbel
precedent: 83/127 mismatched, expected).

uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_soft.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_concat_soft
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (256,)
code_sample_mode = "soft"
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
