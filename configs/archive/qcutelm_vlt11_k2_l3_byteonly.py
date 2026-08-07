"""qcute.qcutelm_vlt11 config: strictest convergence-speed ceiling
baseline — byte-NTP-only, no code hierarchy at all.

Forked from configs/qcutelm_vlt11_k2_l3_identity.py, one step further:
quant_type="identity" still ran the FULL multi-objective loss (level0/1/2
NTP + code_match_loss x3 + e_ntp_loss x3) with discretization removed.
This config (Config.byte_only=True) goes further — it disables the code
hierarchy ENTIRELY: no E_0, no code_pre/quantize/codelm[0], no forecast
substitution into D_0's input (pure teacher-forced real embeddings), no
levels 1+ at all. D_0 trains alone as a plain dense byte LM — structurally
almost identical to bytelm_xs_mtp4_ctx1024 (embed -> N transformer blocks
-> predict), just at tier_d_models[0]=96 instead of bytelm's d_model=256.

Session: "restart and disable all other losses, full unconstrained but
byte ntp loss, fork this config first" — this isolates the "narrow
tier_d_models=96 is under half of bytelm's d_model=256" capacity
hypothesis completely on its own, with ZERO multi-objective competition
and ZERO substitution-corruption confound (both of which the identity-
quant run above still had). If THIS still converges much slower than
bytelm despite being architecturally almost identical modulo width, the
answer is squarely "capacity" (tier_d_models=96 vs 256) and widening it is
the fix. If it converges close to bytelm, the width isn't the bottleneck
and the gap traces back to the hierarchy itself (multi-objective
competition / substitution corruption / quantization) — pointing back to
the byte-NTP-only WARMUP idea (temporary, not permanent) or the identity-
quant result instead.

Ks/dqs/tier_d_models/quant_type/e_ntp_*/code_match_weight are all inert
here (byte_only bypasses everything downstream of D_0) — kept in the
config file for parity/diffability against the other two configs, not
because they do anything.

    uv run python -m qcute.qcutelm_vlt11 --config configs/qcutelm_vlt11_k2_l3_byteonly.py
"""
from pathlib import Path

Ks = (4, 4, 4)
dqs = (8, 8, 8)
tier_d_models = (96, 96, 96)
context_len = 1024
quant_type = "identity"   # inert under byte_only=True — see module docstring
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

code_match_weight = 1.0   # inert
e_ntp_weight = 1.0        # inert
e_ntp_every = 4           # inert
e_ntp_bit_head_mode = "independent"   # inert

byte_only = True   # the actual point of this config — see module docstring

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False
constant_steps = 100
log_every = 100
eval_every = 100
eval_batches = 20
