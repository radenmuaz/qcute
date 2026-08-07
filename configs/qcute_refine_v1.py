"""qcute.qcute_refine config: first training run of the pure recursive
NTP tower + joint-chain-MTP detokenizer (see qcute/qcute_refine.py's
module docstring and docs/qcute_refine_math.md for the full design).

Baseline being matched: configs/bytelm_xs_mtp4_ctx1024.py — context=1024,
dense (unwindowed) attention over the FULL context at every layer/
position by construction, steps=8000, batch_size=16, lr_peak=6e-4,
warmup_steps=500, cosine_decay=False. All five held identical here.
Flagged tradeoff: at this architecture's measured ~1.157 s/step (window=
128/detok_attn_window=64 shape, see below), 8000 steps is ~154 min
(~2.6h) wall-clock — well past this session's earlier informal ~1h
target. Kept anyway, on explicit instruction to match the baseline's
optimizer/step settings exactly rather than truncate steps to fit a time
budget; if that tradeoff is unwanted, override --steps down for a faster,
partial-budget comparison instead.

Architecture: 3 levels, Ks=(2,2,2) (session revision from an initial
Ks=(4,4,4) attempt), dqs=(8,8,8), tier_d_models=(96,96,96),
tier_n_layers=(1,1,1) (the "targets stable" default). seq_lens =
[1024, 512, 256] (context_len halving at each level, not quartering).

attn_window=128, detok_attn_window=64 — chosen (session check) to be the
LARGEST values that still land strictly below every level's own sequence
length (min(seq_lens)=256, min(code_seq_lens)=128), so every level uses
CausalSelfAttention's genuine chunked/windowed path, not the T<=window
dense fallback (an earlier attn_window=256 attempt hit exactly that
fallback at level 2, where seq_len==window — silently correct, since
dense-over-256 and 2-chunk-windowed-over-256 are mathematically
equivalent there, but not the intended "genuinely windowed" mechanism,
and prints a warning). Concretely, with window=128:
  - level 0 (seq=1024, own unit=1 byte): reach = 2*128 = 256 raw bytes.
  - level 1 (seq=512, own unit=K_0=2 bytes/position): reach = 2*128 = 256
    own positions = 512 raw-byte-equivalent.
  - level 2 (seq=256, own unit=K_0*K_1=4 bytes/position): reach = 2*128 =
    256 own positions = 1024 raw-byte-equivalent = full context — reached
    via TRUE 2-chunk windowed attention now, not the dense fallback.
detok_attn_window=64 gives the same "genuinely chunked, not dense-
fallback" property over code_seq_lens=[512,256,128] (gcd=128; 64 is the
largest divisor of all three that's still strictly below the smallest,
128).

Optimizer/LR/steps: fixed identical to the baseline (see above) —
steps=8000, batch_size=16, lr_peak=6e-4, warmup_steps=500,
cosine_decay=False, weight_decay left at Config's own default (1e-5).

Wall-clock check (this session, MPS, fresh init, 15-step timed loop after
3 warmup steps, batch_size=16/context_len=1024, this exact
window/detok_attn_window shape, no fallback warnings observed): 1.157
s/step (0.864 it/s) -> ~154 min for the full 8000-step budget. Re-check
actual logged it/s once underway — this repo's MPS throughput has been
observed to drift over long runs (CLAUDE.md).

    uv run python -m qcute.qcute_refine --config configs/qcute_refine_v1.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v1
"""
from pathlib import Path

Ks = (2, 2, 2)
dqs = (8, 8, 8)
tier_d_models = (96, 96, 96)
tier_n_layers = (1, 1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = 128

detok_d_model = 96
detok_n_heads = 4
detok_n_layers = 2
detok_mlp_mult = 4
detok_attn_window = 64

detok_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20
