"""Same as configs/qcute_refine_v4_4_bpelike_k4_1.py (Ks=(4,1), level0 groups 4-byte blocks,
level1 processes that code sequence at block size 1), PLUS Config.decode_self_only_aux=True.

Why: level0's decode was previously trained on exactly ONE fixed track combination every step
(self + level1's code, "cond_full") -- the "self-only" mode (self track, level1's code dropped)
never got ANY gradient signal, even though it's exactly the regime generate_no_cache silently
falls back to on 3 of every 4 generation steps (see docs/status.md's "generate_no_cache ...
ragged-length conditioning gap" note) and exactly the regime decode ends up in if level1's own
AR code generation degenerates (observed directly: bpelike_k4_1's checkpoint collapsed to a
single repeated level1 code during generation, see docs/status.md's checkpoint verification
section). decode_self_only_aux=True adds an ALWAYS-ON second decode pass every step using only
level0's own code (tracks[:1]), alongside (not instead of) the existing full-cumulative pass --
both contribute to the loss every step now. Reported as level0_ntp_loss_decode_self /
level0_ntp_acc_decode_self in the run log, and decode_self_only_total in the loss breakdown.

qualitative_generate now also prints a third generation mode every eval: level0_cond_self
(forced self-only conditioning at every step, via generate_self_only_cond) alongside the
existing level0_uncond and level0_cond_full -- this run is the first to actually exercise that
third print, and the first with a self-only loss term to observe converging (or not) in
run.log.

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_bpelike_k4_1_selfonly_aux.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_bpelike_k4_1_selfonly_aux/run.log
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = (8, 256)
decode_pack_mode = "interleave"
decode_chunked = False  # decode_K=4 != 1 -- chunked path only implemented/verified for decode_K==1
decode_self_only_aux = True

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
