"""qcute.qcutelm_vlt11 config: full-scale run of the recursive Pass1(E)/
Pass2(D) sandwich, applied 3 levels deep (see qcutelm_vlt11.py's module
docstring for the full rationale and the session's confirmed graph-check).
Ks=(4,4,4) — LOCAL per-level compression factor (not cumulative, unlike
qcutelm_vlt10's periods) — gives seq_lens [1024, 256, 64]: genuine
sequence-length shrinkage at every level, matching qcutelm_vlt7/vlt8's
narrow/wide compute argument applied recursively. dqs/tier_d_models kept
uniform across levels here for a first full-scale run; per-level variation
is sanity-tested in isolation (session CPU smoke test) but not yet tried
at full scale.

attn_window=64 / lm_attn_window=16: the largest values that evenly divide
every level's own (shrinking) sequence length for this Ks/context_len
choice (gcd(1024,256,64)=64, gcd(256,64,16)=16) — see Config's own
docstring for why effective attention reach is ~2*window per layer, not
the full sequence, and why that's fine here (long-range signal is expected
to travel through the code hierarchy). Note: generate_kv_cache requires
attn_window=-1 (dense) — a checkpoint trained with this config's windowed
attention would need attn_window overridden to -1 for KV-cache generation
specifically (generate_no_cache has no such restriction).

    uv run python -m qcute.qcutelm_vlt11 --config configs/qcutelm_vlt11_k2_l3_full.py
"""
from pathlib import Path

Ks = (4, 4, 4)
dqs = (8, 8, 8)
tier_d_models = (96, 96, 96)
context_len = 1024
quant_type = "ifsq"
fsq_levels = 8

n_heads = 4
n_layers = 2
mlp_mult = 4
attn_window = 64

lm_d_model = 128
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4
lm_attn_window = 16

code_match_weight = 1.0
e_ntp_weight = 1.0  # detach-teacher-force fix for the cyclic-target problem —
# see docs/status.md's qcutelm_vlt11 section for the 400-step CPU
# smoke test showing this prevents level1_acc/level2_acc collapsing to
# 100% (codes becoming trivially self-predictable, code_match_loss->0)
e_ntp_every = 4          # amortization strategy 1: only pay for the extra E_i NTP head every 4th
                          # step, not every step — see Config.e_ntp_every's docstring / docs/status.md
e_ntp_bit_head_mode = "independent"  # amortization strategy 3: cheaper head for the aux loss (no
                          # per-bit chain self-attention) — see Config.e_ntp_bit_head_mode's docstring

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False
constant_steps = 100
eval_every = 100
eval_batches = 20
