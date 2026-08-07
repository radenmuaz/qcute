"""qcute.qcute_refine_v1 config: "pure last-layer byte NTP" ablation —
QUEUED, not yet launched (only one training job at a time; run this after
qcute_refine_v1 finishes). Same tower shape as configs/qcute_refine_v1.py
(Ks=(2,2,2), context_len=1024, same optimizer/LR/step budget as
bytelm_xs_mtp4_ctx1024), with three changes:

  quant_type = "identity"   # no BSQ discretization anywhere in the tower
  code_ntp_weight = 0.0     # levels 1/2's own NTP loss: SKIPPED, not just zero-weighted
  detok_weight = 0.0        # every Detokenizer's forward: SKIPPED, not just zero-weighted

Net effect: total loss reduces to EXACTLY level 0's own byte NTP loss
(verified: `loss == byte_loss` bit-for-bit in a smoke test) — levels 1/2
and all 3 detokenizers still run their own trunk forward (codes still get
computed and handed up the tower, since level i+1's input is level i's
code by construction) but contribute zero gradient anywhere, and their
own expensive per-level heads (BitPredictHead in chain mode) are never
even called. This isolates two questions at once:

  1. Quality: does qcute_refine's byte-level performance on its own
     (no code-level or detokenizer supervision competing for gradient,
     same spirit as qcutelm_vlt11's byte_only diagnostic) differ from
     the full multi-objective qcute_refine_v1 run, or from bytelm itself?
  2. Speed: session diagnosis (CPU microbenchmark) found BitPredictHead's
     chain mode ~200x (dq=8) to ~1800x (dq=32) slower per call than a
     plain nn.Linear at the same batch size, scaling worse than linearly
     in dq — and every Detokenizer's own mtp_head has dq = K*in_dq
     (bigger than any encoder ntp_head's dq = in_dq alone), making it the
     single most expensive component in the whole tower. Skipping ALL of
     that (code_ntp_weight=0.0 skips 2 of 3 encoder chain calls,
     detok_weight=0.0 skips all 3 detokenizer chain calls — 5 of the
     original 6 per-level chain-head calls gone, leaving only level 0's
     own byte ntp_head) should measurably close qcute_refine's ~13-15%
     it/s gap vs the dense-attention bytelm_xs_mtp4_ctx1024 baseline
     (0.86 vs 0.98 it/s, this session's own measurement) — worth
     confirming once this run is underway (compare its own logged it/s
     against qcute_refine_v1's).

    uv run python -m qcute.qcute_refine_v1 --config configs/qcute_refine_v2_pure_byte_ntp.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v2_pure_byte_ntp
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

quant_type = "identity"
code_ntp_weight = 0.0
detok_weight = 0.0

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
