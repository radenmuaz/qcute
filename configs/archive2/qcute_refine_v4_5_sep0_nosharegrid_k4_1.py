"""2x2 grid cell (v4.5 x 2-level), per user request: share_level_weights=False. Under v4.5's
staged cross-attention design this means, per level: encode LM (reused, unchanged, for decode's
own Stage 0 -- no separate weights there at all) + one INDEPENDENT LM per cross-attention track
(not one shared decode LM per level) -- level0 here gets 2 independent cross-attn-stage LMs (self
code, then level1's code), level1 (topmost) gets 1. See qcute_refine_v4_5.py's Config.
share_level_weights and RefineLM._run's decode_stage_lms for the full design. Every non-final
stage's own NTP loss is summed into the total loss (decode_stage_extra_total in the metrics) --
otherwise those independent weights would only get weak indirect gradient.

Combined with the session's other established "good settings": use_gumbel_noise=True +
gumbel_tau=2.0, cross_track_source="decode" + decode_code_ste=False. Same overfit10k testbed
(n_bytes=10000, steps=1000) as the rest of this session's batch.

Companion grid cells:
  - qcute_refine_v4_5_nosharegrid_k1.py    (v4.5, 1 level)
  - qcute_refine_v4_4_nosharegrid_k4_1.py  (v4.4, 2 level)
  - qcute_refine_v4_4_nosharegrid_k1.py    (v4.4, 1 level)

    uv run python -m qcute.qcute_refine_v4_5 --config configs/qcute_refine_v4_5_sep0_nosharegrid_k4_1.py

    # watch live:
    tail -f logs/qcute_refine_v4_5_sep0_nosharegrid_k4_1/run.log


decode_separate_stage0=True variant (see qcute_refine_v4_5.py's Config docstring): decode's Stage 0 no longer reuses encode's own h -- gets its own independent LM instead, fully decoupling encode's training from decode's gradient. Rerun of this same grid cell to test whether that removes v4.5's generation-collapse problem."""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = (8, 256)
cross_track_source = "decode"
decode_code_ste = False
share_level_weights = False
decode_separate_stage0 = True
use_gumbel_noise = True
gumbel_tau = 2.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
qual_prompt_bytes = 64
