"""Ablation, per explicit user request: cross_track_source="decode" instead of the default
"encode". Same base numbers as configs/qcute_refine_v4_4_bpelike_k4_1.py (Ks=(4,1), d_model=256,
n_layers=2, context_len=256, attn_window=(8,256)) -- the only change is where level0's cross
track (level1's code, used as one of decode's conditioning inputs) is sourced from.

Default ("encode"): level1's code comes from its plain UNCOND ENCODE pass (c_list[1] -- pure
self-attention over code_0, no code-conditioning applied). This config ("decode"): level1's code
instead comes from its OWN COND (self-conditioned) DECODE pass -- LevelLM.forward already computes
a fresh code from decode's packed/conditioned hidden state via the same pooling+classify+quantize
pipeline as encode, previously discarded by RefineLM._run, now captured and used.

User's stated rationale: decode is reconstruction-from-latent (code -> embed -> reconstruct),
which can be detached, and recursively generating each level's code from ITS OWN decode pass is
more architecturally consistent with that generative structure than pulling from an uncond-NTP-
focused encode pass. Requires top-down decode order (level1 decodes first, level0 second) so
level0 can read level1's already-computed decode-derived code -- RefineLM._run now always
iterates top-down (harmless for the default "encode" case, required for this one). Self tracks
are UNAFFECTED either way (always source from encode -- a level can't condition its own decode on
its own not-yet-decoded output); only the CROSS track changes. Does not apply to n_levels==1
configs at all (no cross track exists there).

decode_code_ste=False (detach, NOT the default) -- explicit follow-up per user direction ("with
detach"), pairing cross_track_source="decode" with the same detach principle already established
for the self-conditioning experiments: the decoder should condition on a code as a fixed discrete
query/embedding lookup (latent-variable / Markov-chain style), not as a soft mixture whose
gradient could reshape the very latent it's conditioning on. Without this, decode's gradient could
flow back through the RECURSIVELY-SOURCED decode-derived code into level1's OWN code producer,
compounding across levels in a way that's especially hard to reason about now that decode's own
output feeds the next level's conditioning input -- detach keeps that boundary clean the same way
it does for the (already-tested) self-conditioning-only design.

What to compare against bpelike_k4_1's own results: val_level0_ntp_acc_decode/val_bpb (does
decode quality change), and scripts/probe_code_usage_entropy.py's code_0/code_1 entropy (does
recursively-generated conditioning change the collapse pattern investigated earlier this session).

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_bpelike_k4_1_crosstrack_decode.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_bpelike_k4_1_crosstrack_decode/run.log
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = (8, 256)
decode_pack_mode = "interleave"
decode_chunked = False
cross_track_source = "decode"
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
