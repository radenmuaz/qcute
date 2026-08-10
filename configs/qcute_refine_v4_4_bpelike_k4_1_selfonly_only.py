"""Same base numbers as configs/qcute_refine_v4_4_bpelike_k4_1.py (Ks=(4,1), d_model=256,
n_layers=2, context_len=256), but level0's decode conditions on ONLY its own code (self track) --
the cross track to level1's code is disabled entirely (decode_window=0 for that source), not just
de-prioritized. No decode_self_only_aux needed here: since the cross track never gets built at
all (excluded in RefineLM._run's track-gathering loop before decode ever runs), the ONE decode
pass level0 does run IS the self-only pass -- "no ntp on cond self+code1" by construction, not by
adding a second parallel pass alongside a first one (that's what decode_self_only_aux does, and
is a different, already-tested config: bpelike_k4_1_selfonly_aux).

Why this run exists: qcute_refine_v4_4_bpelike_k4_1's own checkpoint (and, confirmed live during
bpelike_k4_1_selfonly_aux's own qualitative output at step ~2500) showed cond_full's code
collapsing to a SINGLE repeated value (all {24} in one observed sample) and level1_gen ALSO
collapsing to a single repeated code, while that SAME run's cond_self showed healthy, varied
codes. This isolates whether dropping the cross-level dependency ENTIRELY (not just adding a
parallel self-only signal) produces a model whose only generation mode -- since level0 never had
any other mode trained in the first place -- is healthy, i.e. whether the cross-track/level1
dependency is *the* cause of the collapse, not just *a* contributing factor.

Level1 itself is left otherwise unchanged (Ks=(4,1) unchanged, level1 still trains its own encode
NTP over the code_0 stream) so its own qualitative output stays available for comparison, even
though level0 no longer depends on it for anything.

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_bpelike_k4_1_selfonly_only.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_bpelike_k4_1_selfonly_only/run.log
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = ((8, (8, 0)), 256)  # level0: encode=8, decode=(self=8, cross=0/disabled); level1: 256 (dense)
decode_pack_mode = "interleave"
decode_chunked = False

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 4000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20

qual_gen_bytes = 64
qual_prompt_bytes = 64
