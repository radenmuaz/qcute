"""qcute.qcutelm_vlt6 config: shared_encoder_decoder=False — encoder and
ZERO-KV ABLATION (use_zero_kv=True) — see other docstring text below for the rest of this cell's
setup (quant/loss/encdec); this variant additionally re-enables the zero-KV sink (paired with RoPE,
a combination not used in the original design) to isolate use_zero_kv as its own grid dimension.

decoder get fully independent weights (own byte_emb/blocks/ln_f/head/
z_proj) instead of the original design's single shared Block stack reused
for both the byte-level encoder pass and the code-as-BOS decode pass.
Symmetric by default (dec_d_model/dec_n_heads/dec_n_layers/dec_mlp_mult
all None here, so the decoder mirrors the encoder's own d_model=96,
n_heads=4, n_layers=2, mlp_mult=4 exactly — genuinely separate weights,
not a smaller/larger split) — override any dec_* field to make it
asymmetric instead.

NOT YET RUN — queued per "try this later". Same RoPE+no-zero-KV+windowed
scheme and loss weights as qcutelm_vlt6_rope_bpelm_parity.py otherwise,
so this isolates shared_encoder_decoder as the sole variable against that
run once both have comparable data.

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_grid_ifsq_ntp_encdec_zerokv.py
"""
from pathlib import Path

K = 4
context_len = 1024
attn_window = 64
dq = 5
quant_type = "ifsq"
fsq_levels = 8

d_model = 96
n_heads = 4
n_layers = 2
mlp_mult = 4
code_net_layers = 0

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4

use_rope = True
use_zero_kv = True

shared_encoder_decoder = False  # symmetric by default — dec_* fields left unset, mirrors encoder dims

main_ntp_weight = 1.0
aux_recon_weight = 0.0
code_match_weight = 0.0

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

gen_every = 1000
gen_prompt_len = 64
gen_new_bytes = 64
