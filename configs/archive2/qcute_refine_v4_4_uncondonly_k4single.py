"""v4.4 twin of qcute_refine_v4_5_uncondonly_k4single.py -- structural-bug diagnostic.

Motivation (user's direct suspicion): LevelLM.encode() (v4.5) and v4.4's plain
`decode_tracks is None` branch were verified byte-identical in code (same rope call, same block
loop, same final ln_f). Under decode_ntp_weight=0.0, decode contributes exactly zero gradient in
BOTH architectures -- so encode_lms's training trajectory should be, structurally, the same
uncond byte-LM problem either way. If v4.4's uncondonly run here generates cleanly while v4.5's
twin (qcute_refine_v4_5_uncondonly_k4single.py) stays collapsed under the exact same data/steps/
hyperparams, that is real evidence of a v4.5-specific bug (most likely in generation-time code --
generate_no_cache/generate_encode_only/_sample_next_byte/final_embed_weight plumbing -- since the
training-loss code path was already checked and found identical). If v4.4 ALSO collapses the same
way, that confirms the original "ordinary exposure bias, not a bug" conclusion instead.

Identical hyperparams to the v4.5 twin: share_level_weights=False, decode_code_ste=False,
use_gumbel_noise=False, decode_ntp_weight=0.0, single-level Ks=(4,), n_layers=2, overfit10k testbed.

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_uncondonly_k4single.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_uncondonly_k4single/run.log
"""
from pathlib import Path

Ks = (4,)
d_model = 256
n_layers = 2
context_len = 256
attn_window = ((8, 256),)
decode_code_ste = False
share_level_weights = False
use_gumbel_noise = False
gumbel_tau = 1.0
decode_ntp_weight = 0.0

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
