"""qcute.qcutelm_vlt config: variable-length causal tokenizer
(qcute/qcutelm_vlt.py) — 4-layer encoder AND 4-layer decoder (both causal
transformers, zero-KV self-attention, NoPE).

Progression: d_model=44 (~1.0x corpus-bit parity) underfit at T=4 within a
20000-step budget; d_model=128 (~7.3x parity) was judged too big; settled
on d_model=64 (433,490 params -> 13,871,680 bits -> ratio 1.93x corpus
bits — roughly double d_model=44's params, not literally double d_model).

Curriculum now requires actually clearing --curriculum_target_acc (95%
val recon_acc) before advancing a stage — curriculum_max_steps_per_stage
raised high enough that it's a safety net against a truly stuck stage,
not the normal advance path (T=2 previously advanced via the step
fallback at only 79.69% val, a real train/val gap — see docs/status.md-
style session notes). Also now early-stops once the final stage (T=K)
itself clears the target, instead of always running to --steps.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm_vlt --config configs/qcutelm_vlt_4layer.py
"""
from pathlib import Path

K = 4
d_model = 64  # up from 44 (~1.0x corpus bits) -> ~1.93x; 128 (~7.3x) tried and judged too big
n_heads = 4
n_layers_enc = 4
n_layers_dec = 4
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 100000  # longer budget — each stage must now actually hit target_acc, not just survive a step count
batch_size = 16
lr_peak = 1e-3
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = True
constant_steps = 100
eval_every = 100
eval_batches = 20

curriculum_target_acc = 0.95
curriculum_max_steps_per_stage = 30000  # safety net only — target_acc should drive advancement, not this
