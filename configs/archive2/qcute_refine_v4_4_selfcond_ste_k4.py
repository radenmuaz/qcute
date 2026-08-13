"""Ablation counterpart to configs/qcute_refine_v4_4_selfcond_detach_k4.py: identical in every
way (same self-conditioning-only design, level1 as pure NTP drafter, level1's own decode
disabled) EXCEPT decode_code_ste=True (the default -- straight-through, decode's loss backprops
into level0's own code producer), instead of False (detach).

Why this ablation matters: decode_code_ste=False was motivated by docs/two_stage_latent_decode_
math.md's requirement that level1's independent NTP training target (code_0) stay undistorted by
decode's needs -- but that requirement was never actually A/B tested against the straight-through
alternative, because decode_code_ste=False silently never took effect at all until this session's
main()/argparse wiring bug fix (see docs/status.md) -- selfcond_detach_k4's ORIGINAL run
unintentionally WAS this ablation's condition (STE) despite its name and docstring. Now that the
wiring is fixed and a genuine detach rerun exists (selfcond_detach_k4_rerun), this config supplies
the missing other half of the comparison: same architecture, same everything, ste vs detach, to
directly test whether the detach requirement actually matters here (does code_0's own quality --
val_level0_ntp_acc_decode/val_bpb -- differ meaningfully between the two, and does level1's own
drafting accuracy at predicting code_0 -- val_level1_ntp_acc_encode -- differ, e.g. because STE's
extra gradient path into the code producer could pull code_0 in a direction level1's NTP head
struggles to track, or conversely could help by keeping code_0's distribution less degenerate).

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_selfcond_ste_k4.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_selfcond_ste_k4/run.log
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = ((8, (8, 0)), (64, 0))
decode_pack_mode = "interleave"
decode_code_ste = True

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
