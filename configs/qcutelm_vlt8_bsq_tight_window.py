"""qcute.qcutelm_vlt8 config: tight-window ablation, testing whether the
code_conditioned_acc/no_code_acc gap (flat/near-identical across the
entire qcutelm_vlt7_bsq / qcutelm_vlt8_bsq runs so far — e.g. 52.67% vs
53.06% at step 3938 of the v7 run) is a real finding about code value, or
an artifact of the window being far wider than K.

Session analysis: attn_window=64 (v7) / 80 (v8's block-aligned version,
16*(K+1)) lets the chunked-attention mechanism reach ~2*window raw
positions of true history (~32 blocks at window=80) — vastly more than
the K=4/one-block granularity a code is supposed to compress. Both Pass 1
("no-code") and Pass 2 ("forecast") run through the SAME windowed stack,
so with a window this wide, Pass 1 can reconstruct almost as much as
Pass 2 just by reading raw bytes still inside its window — the code
becomes redundant with information already available for free, not the
load-bearing channel qcutelm_vlt6's decode_block (architecturally
block-local, ZERO cross-block attention, see session notes) guaranteed
by construction.

attn_window=K+1=5 (m=1, the tightest legal value under qcutelm_vlt8's
block-alignment invariant) closes most of that shortcut: each chunk is
exactly one block, with the chunked mechanism's "attend to previous
chunk" rule giving at most one block of raw-byte lookback beyond the
current one (not zero, unlike qcutelm_vlt6's decode_block — a known
residual gap, see session notes) — as close to qcutelm_vlt6's clean
separation as this architecture can get without also masking out the
previous-chunk term entirely. If code_conditioned_acc pulls meaningfully
ahead of no_code_acc here where it didn't at window=80, that confirms the
wide window was masking the code's real contribution.

Otherwise identical to qcutelm_vlt8_bsq.py — same bsq/dq=13/d_model=96/
lm_d_model=256/context_len=1024 architecture.

    uv run python -m qcute.qcutelm_vlt8 --config configs/qcutelm_vlt8_bsq_tight_window.py
"""
from pathlib import Path

K = 4
context_len = 1024
dq = 13
quant_type = "bsq"
fsq_levels = 8

d_model = 96
n_heads = 4
n_layers = 2
mlp_mult = 4
attn_window = 5  # K+1, m=1 — tightest legal block-aligned window, minimal raw-byte leak beyond the current block

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4

code_match_weight = 1.0
shared_tokenizer_phases = True  # pinned explicitly — isolates attn_window as the only variable
                                 # against qcutelm_vlt8_bsq.py (which also used True); the untied-by-
                                 # default change happened after this config was written

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
