"""Two-stage latent-variable decode: self-conditioning + .detach() (no STE) + level1-as-drafter.
See docs/two_stage_latent_decode_math.md for the full derivation this config exists to test.

Ks=(4,1): level 0 self-conditions on its OWN code (code_0) for decode -- the cross-level track
to level1's code is explicitly DISABLED (attn_window's decode_window for level0's +1 source is 0),
so level1 contributes NOTHING to decode's training signal. level1 exists purely as an independent
NTP model over the code_0 stream (its own encode_losses[1], via the standard encode pass) -- this
IS the "drafter" from the math doc, reusing generate_level1_codes unmodified, no new machinery
needed. level1's OWN decode/self-conditioning pass is also disabled (its attn_window's
decode_window is 0) since it's irrelevant to this experiment's question.

decode_code_ste=False (REQUIRED, not the default): decode's loss must NOT backprop into level0's
own code producer, so level1's independent NTP training target (code_0) stays undistorted by
decode's needs -- verified directly: enc0.code_head.weight.grad is None when isolated to
decode_losses[0].backward() alone.

What this config is FOR: after training, compare level1's own NTP accuracy at predicting code_0
(val_level1_ntp_acc_encode) against decode's own quality (val_level0_ntp_acc_decode/val_bpb) --
if level1 predicts code_0 well, its drafted continuation (generate_level1_codes) is a candidate
substitute for decode's true self-conditioning code_0 at generation time, per the math doc's
Sec.7 draft-substitution scheme (NOT implemented as a generation function yet -- this config only
tests whether the TRAINING setup produces high-quality-enough drafts to make that worthwhile).

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_selfcond_detach_k4.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_selfcond_detach_k4/run.log
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = ((8, (8, 0)), (64, 0))
decode_pack_mode = "interleave"
decode_code_ste = False

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
