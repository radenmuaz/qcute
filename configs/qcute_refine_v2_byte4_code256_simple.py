"""qcute.qcute_refine_v2 config: same architecture as
configs/qcute_refine_v2_byte4_code256.py (single byte-LM + single code-LM
tower, K=4, context_len=1024, d_model=256 matched to bytelm), but with
BitPredictHead removed from the encoder tower entirely (session: "for
this run, simplify first not using bitpredicthead"):

  byte_repr = "embed"        # level 0: traditional nn.Embedding(vocab,D)
                              # lookup + plain nn.Linear(D,vocab) NTP head,
                              # 256-way cross-entropy — exactly bytelm.py's
                              # own convention (both flags kept as real
                              # options, not removed — see Config's own
                              # docstring in qcute_refine_v2.py).
  code_head_mode = "independent"   # level 1: single plain nn.Linear(D,dq),
                              # independent per-bit logits, no chain-rule
                              # cross-bit conditioning — "code level use
                              # independent linear heads like original bsq."

DecoderLevel's own tok_head_mode stays "linear" (already the default,
already BitPredictHead-free) — so with both flags above set, this run
uses BitPredictHead nowhere in the model at all (verified: 0 instances).

Everything else — Ks=(4,4), context_len=1024, per-level attn_window=
(256,64) (genuinely windowed at both levels, not a coincidental dense
fallback), tier_d_models=(256,256), optimizer/LR/steps matched to
bytelm_xs_mtp4_ctx1024 — is unchanged from qcute_refine_v2_byte4_code256.py.

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_v2_byte4_code256_simple.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v2_byte4_code256_simple
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (256, 64)

byte_repr = "embed"
code_head_mode = "independent"

tok_d_model = 256
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

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
