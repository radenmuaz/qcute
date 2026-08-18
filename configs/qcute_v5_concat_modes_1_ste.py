"""qcute_v5_concat_modes_1_ste: configs/qcute_v5_concat_modes_1.py with code_sample_mode="ste"
instead of "soft" -- isolates whether multi_mode_impl="single_pass" (Ks=(4,1), level0 gets a
self-only mode1 loss alongside the deepest modefull) is a workable decoder architecture on its own,
independent of soft mode's known issues (127/127 gen_consistency mismatches from Gumbel-noise RNG
divergence between teacher-forced and incremental generation calls; qualitative bpb/legibility
divergence between mode1 and modefull -- see docs/status.md's 2026-08-18 entry). Under ste
(deterministic hard forward, no noise), check_gen_consistency should be genuinely meaningful (0
mismatches expected if multi-mode decode is implemented correctly), and mode1 vs modefull
qualitative comparisons aren't confounded by stochastic sampling. Everything else unchanged from
qcute_v5_concat_modes_1.py (same Ks/d_model/context_len/attn_window/steps/schedule) so this is a
clean single-variable ablation.

uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_modes_1_ste.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_concat_modes_1_ste
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_sample_mode = "ste"
multi_mode_impl = "single_pass"

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 2000
eval_batches = 20

qual_gen_bytes = 128
qual_prompt_bytes = 64
