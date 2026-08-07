# Status

Tracks progress on the **active** lineage — `qcute/qcute_refine_v1.py` /
`qcute/qcute_refine_v2.py` — going forward from this session onward.

**Everything prior** (Phase 0-3 of the original `continuous_tokenizer_
handover.md` plan, and the full `qcutelm`/`qcutelm_vlt`/`qcutelm_pyramid`/
`qcute_fifo`/`qcute_bytepool` fork-by-fork narrative that plan produced) is
archived at [docs/archive/status_archive.md](archive/status_archive.md) —
that lineage itself is archived under `qcute/archive/`
(`configs/archive/`), superseded by `qcute_refine`. Read the archive for
historical context/reproducibility; nothing there is still being acted on.

`bytelm.py`/`bpelm.py` remain the active baseline comparison points (not
archived) — see `docs/archive/status_archive.md` for their own original
setup narrative, still valid.

## Baseline numbers (current reference point for every `qcute_refine` comparison)

All 8000-step runs, batch_size=16, `datasets/enwik8_1M.gz`. it/s = mean
over the full logged run (not just the best region). Best val_bpb/step
computed from each log's final (non-stale) segment.

| run | context | best val_bpb | @ step | mean it/s | wall time (8000 steps) |
|---|---|---|---|---|---|
| `bytelm_xs_mtp4_ctx1024` | 1024 bytes | **2.365** | 1700 | 1.13 | 2:27:31 |
| `bpelm_8192` | 256 tok (~845 byte-equiv) | **2.350** | 800 | 4.23 | 0:34:25 |
| `bpelm_32768` | 256 tok (~973 byte-equiv) | **2.134** | 500 | 2.08 | 1:07:41 |
| `qcute_refine_v2_byte4_code256_simple` ("v1") | 1024 bytes | **2.485** | 5600 | 2.42 | 1:00:53 |

`qcute_refine_v2`'s "v1" run: worse best-bpb than either bpelm variant,
better than plain `bytelm`, but at ~2.1x `bytelm`'s throughput and ~4.3M
fewer params (2.706M vs. bytelm's 3.412M) — see the FLOPs/param
comparisons earlier in this session's conversation log for the fuller
picture (not yet transcribed into this file). Every baseline here
overfits well before 8000 steps (see the step-budget finding below) —
these best-bpb numbers, not the final-step ones, are the actual
comparison target.

## `qcute/qcute_refine_v1.py` / `qcute/qcute_refine_v2.py` — new fork lineage (this session)

A second, independent fork lineage alongside `qcutelm_vlt*`/`qcutelm_pyramid*`
(self-contained-module convention, same as the rest of this project — full
per-version design rationale lives in each file's own module docstring, not
duplicated here; this entry is a pointer + one cross-cutting finding, not a
full recap). `qcute_refine_v1.py`: pure recursive NTP
tower with BSQ code hand-off between levels, plus a block-local joint-chain-
MTP detokenizer. `qcute_refine_v2.py`: detokenizer redesigned into a
`DecoderLevel` that cross-attends between adjacent levels' own
`EncoderLevel` hidden states (reused, not recomputed) instead of running a
block-local self-attention pass; grew a number of session-driven flags
(`byte_repr`, `code_head_mode`, `bit_head_class` with `BitPredictHeadAttn`/
`Conv`/`SSM` variants, `cross_attn_rope`, `decoder_own_trunk`,
`decoder_kv_pass_through`/`decoder_q_pass_through`, `layer_warmup_steps`)
plus a real MPS-specific bug fix (`nn.MultiheadAttention`'s backward
produced NaN gradients at `d_model=256`, resolved by switching to manual
`F.scaled_dot_product_attention` throughout, matching every other attention
op in the file). Configs live under `configs/qcute_refine_v2_*` and
`configs/qcute_refine_*`.

**Baseline step-budget finding, applies project-wide, not just to this
fork**: checked `bytelm_xs_mtp4_ctx1024`'s and `bpelm_32768`'s own full
8000-step val_bpb curves (excluding each log's earlier stale/restarted
segment). Neither one benefits from the full 8000-step budget —
`bytelm_xs_mtp4_ctx1024` bottoms out at **step 1700 (val_bpb 2.365)** then
overfits almost monotonically to 4.43 by step 8000, no flat region at all;
`bpelm_32768` bottoms out at **step 500 (val_bpb 2.134)**, overfits sharply
through ~step 3000-4000, then genuinely plateaus (noisy, no further trend)
through 8000. Since both baselines are already fully into their
overfit/plateaued state by step 4000, comparison runs gain no signal from
the second half of an 8000-step budget — **new `qcute_refine_v2` ablation
configs default to `steps=4000`** going forward (already applied to every
queued-not-yet-launched config as of this session: `qcute_refine_rope`,
`qcute_refine_decoder_trunk`, `qcute_refine_pass_through`,
`qcute_refine_v2_byte4_code256_identity`, `qcute_refine_rope_3level_curriculum`).
Also worth adopting project-wide: report **best-checkpoint val_bpb**
(`checkpoints/<run>/best.pt`, already tracked by `Checkpointer`), not
final-step val_bpb, as the headline comparison number — final-step numbers
on these small-corpus runs mostly measure how overfit a run got, not how
good its best state was.

Documentation note: `CLAUDE.md`'s own Commands section previously pointed
its `bytelm` example at `configs/bytelm_xs_mtp4.py` (`context=256`) —
updated to `configs/bytelm_xs_mtp4_ctx1024.py`, the actual standard
baseline as of this session (`context=1024`, matching `qcute_refine`'s own
`context_len`). The old `context=256` config is kept (historical
reproducibility) but is no longer the comparison target for new work.

## `qcute_refine_v2_byte4_code256_simple` ("v1") finished — plateaus, doesn't overfit catastrophically

Full 8000-step run completed (`byte_repr="embed"`, `code_head_mode=
"independent"`, no `BitPredictHead` anywhere, `Ks=(4,4)`,
`tier_d_models=(256,256)`, matched to `bytelm_xs_mtp4_ctx1024`'s own
budget). `logs/qcute_refine_v2_byte4_code256_simple/bpb.png`: val_bpb
drops to ~2.6 by step ~1500-2000, then stays flat/noisy (2.5-2.9 band)
the rest of the way to 8000 — unlike `bytelm_xs_mtp4_ctx1024`'s own
monotonic-overfit curve (see archive), this run genuinely plateaus rather
than climbing. `best.pt` landed at **step 5600, val_bpb 2.4846**.

**Decoder KV contribution — probed, genuinely mixed/inconclusive.**
`scripts/probe_decoder_kv_contribution.py` (new this session — gradient-
norm ratio, KV-ablation loss/acc delta, null-slot attention mass; three
independent signals since no one alone is trustworthy) against `best.pt`:
`grad_ratio_curr_over_prev` **0.01-0.02** (KV side's gradient is ~1-2% of
Q side's), `null_slot_attn_mass` **~0.29** (vs. ~0.004 uniform over 257
KV positions — the model has learned to substantially opt out of
attending to real code content), and on val data specifically, **ablating
KV entirely (forcing null-only) *lowers* loss** (1.687 vs. 1.731 with
it) despite accuracy being very slightly *worse* without it (0.532 vs.
0.538). Cross-checked against the FULL `run.jsonl` trajectory (not just
one checkpoint): `pair0_tok_acc` (with KV) is consistently $\ge$
`level0_ntp_acc` (without KV) from step ~1600 onward, small but
persistent (+0.001 to +0.016) — so KV *does* give a real, if small,
accuracy edge across most of training. Read together: **overfit
calibration, not overfit accuracy** — the decoder's cross-attention head
grows more confident in ways that help top-1 accuracy marginally but hurt
held-out cross-entropy (a classic overconfidence signature, not a "KV is
useless" one). Genuinely unresolved which effect dominates in practice;
`qcute_refine_pass_through`'s results (both Q and KV stripped to direct
embeddings/projections, no `h` reliance on either side) are queued
specifically to help resolve this.

**Real bug found and fixed: `DecoderLevel`'s cross-attention KV window
was unbounded.** The causal mask (`b < n_complete(t)`) enforced *that* a
KV block must be complete before being visible, but never capped *how
far back* — every completed block stayed reachable regardless of
distance, inconsistent with the encoder's own windowed self-attention at
that same level (`Config.attn_window`, e.g. `(256,128,64)`). Fixed:
`DecoderLevel` now also takes `kv_window = Config.attn_window[level+1]`
and requires `b >= n_complete(t) - kv_window` too (`None`/`-1` preserves
the original unbounded reach) — see `docs/qcute_refine_math.md` §7.1 for
the full algorithm. Applies automatically to every already-written config
using per-level windows (the whole `qcute_refine_*`/`byte4_code256*`
family) — none had launched yet when this was found, so nothing needed
re-running.

**`Config.cross_attn_rope`** (default `True`) also added this session:
Q gets its own raw-byte-time RoPE position (`0..L-1`); each KV slot gets
the raw-byte-time position it becomes causally resolved at
(`(b+1)*K-1`, or `0` for the null slot) — gives the cross-attention actual
relative-distance information instead of only the boolean visible/blocked
mask it had before. `False` restores the original position-blind
cross-attention. See `docs/qcute_refine_math.md` §7.2.

**Housekeeping**: every earlier qcute-lineage fork (`qcutelm.py`,
`qcutelm_vlt*.py`, `qcutelm_pyramid.py`, `qcutelm_mergetoken_v1.py`,
`qcute_bytepool.py`) archived to `qcute/archive/` (configs to
`configs/archive/`, their own design docs — `continuous_tokenizer_
handover.md`, `fifo_v2.md`, `vlt12_math.tex` — to `docs/archive/`), 93 old
log directories cleared, 4 scripts' broken imports fixed
(`qcute.archive.*`). `bytelm.py`/`bpelm.py` are the explicit exception,
still active baselines. The `v1_*` ablation configs renamed to
`qcute_refine_*` for naming consistency with the rest of the lineage; the
one already-running job (`v1_pass_through`) was stopped and relaunched
fresh under its new name (`qcute_refine_pass_through`) rather than left
mismatched against its renamed config file. `qcute/qcute_refine.py`
itself later renamed to `qcute/qcute_refine_v1.py` (configs/docs updated
to match) for naming consistency with `qcute_refine_v2.py`. This file
itself (`docs/status.md`) reset to just the `qcute_refine`-relevant
entries above — everything prior moved to `docs/archive/status_archive.md`.

## `qcute_refine_pass_through` finished — KV contribution now UNAMBIGUOUS (opposite of "simple"'s inconclusive read)

Full 4000-step run (`decoder_kv_pass_through=True` + `decoder_q_pass_through=True`,
`cross_attn_rope=True`, windowed KV) completed: **best val_bpb 2.5575 @ step
3300** (final-step val_bpb 2.6009, mildly overfit past that point) — worse
than v1's own best (2.4846 @ step 5600, full h_prev/h_curr reuse), so
stripping both Q and KV to direct raw embeddings/projections (no encoder
hidden-state reliance on either side) costs real accuracy vs. reuse mode.

`scripts/probe_decoder_kv_contribution.py` **adapted** this session to
support non-default decoder modes (`_compute_qkv` now mirrors
`DecoderLevel.forward`'s own own_trunk/q_pass_through/kv_pass_through
dispatch instead of hardcoding `h_prev`/`h_curr` reuse — the original
script crashed outright against this checkpoint, since `q_embed` is an
`nn.Embedding` and pass-through mode feeds it raw byte ids, not a
continuous `h_prev` hidden state). Gradient-norm signal (signal 1) is
skipped (`NaN`) on any side using pass_through/own_trunk, since there's no
continuous encoder hidden state on that side to take a gradient w.r.t. —
ablation and attention-mass (signals 2/3) remain fully meaningful and are
the ones that matter here anyway.

Re-run against `qcute_refine_pass_through`'s `best.pt` (8 batches x 16,
train + val): **`delta_loss_from_kv` ≈ −0.23 to −0.24** (ablating KV
entirely *raises* loss substantially — opposite sign from the "simple"
run's val-only finding), **`delta_acc_from_kv` ≈ +0.036 to +0.038**
(consistent, positive, same sign on train AND val — no train/val split
this time), and **`null_slot_attn_mass` ≈ 0.05–0.06** (vs. ~0.29 in the
reuse-mode run — attention is overwhelmingly on real code content, not
opting out to the null fallback). Read together with the "simple" run's
own mixed result: when Q/KV are reused encoder hidden states already
carrying most of the signal, cross-attention has less left to add and
drifts toward overconfident-but-not-more-accurate null-heavy attention
(the earlier "overfit calibration, not accuracy" read); when Q/KV are
stripped to raw pass-through with no such shortcut, the model has no
alternative but to genuinely rely on the cross-attention KV, and does —
consistently, causally, on both loss and accuracy. Net: KV contribution
depends heavily on what else is available to lean on, not a fixed
property of the cross-attention mechanism itself.

## BitPredictHead speed: linear vs. attn/conv/ssm, matmul reparam, and inner-downsample (this session)

`scripts/bench_bit_heads.py` (new): CPU, tiny-scale (`d_model=32`,
`N=32`) forward and forward+backward wallclock for the "independent"
`nn.Linear` baseline (no chain-rule conditioning) vs. the three chain
heads (`bit_head_class` = `attn`/`conv`/`ssm`), across `dq` in
`(8,16,32)`, plus a `decode` mode (`true_bits=None` → `_forward_loop`,
the actual sequential-loop autoregressive-generation-time cost, distinct
from the batched teacher-forced `_forward_fixed` path used for training).

**Train (fwd+bwd) slowdown vs. linear**: all three chain heads cost real
overhead (expected — they pay for cross-bit conditioning the linear head
skips entirely), roughly attn 11-19x, conv 9-51x, ssm 12-32x depending on
dq. **Decode (sequential loop) slowdown is far worse and roughly
dq-scaling**: attn 165-730x, ssm 54-136x (`ssm` consistently cheapest —
no dq-dependent op, just a per-step linear-decay update), and `conv`
worst-case **~3900x at dq=16 specifically** with the original `nn.Conv1d`
implementation — a backend-quirk spike (dq=8 and dq=32 are far cheaper,
~25-72x), not a real algorithmic cost.

**Fix**: `BitPredictHeadConv` gained `conv_impl` (`"matmul"`, new
default, vs. `"conv1d"`, original) — mathematically the SAME operation
(fixed causal window, weights shared across positions), just
reparametrized as `nn.Linear(kernel_size*D, D)` over a flattened window
instead of calling `nn.Conv1d` directly. Verified numerically
fixed/loop-consistent (~1e-7 diff) at every kernel size tried. Fixes the
dq=16 anomaly entirely and is faster across the board: dq=16 train
428x→14x slower than linear, decode 3876x→104x. Wired through
`Config.bit_conv_impl` + CLI; kept BOTH modes as a flag, not a
replacement (repo convention).

**Also added**: `Config.bit_inner_downsample` (1/2/4, default 1 = exact
prior behavior, no extra op/params) — projects the incoming hidden vector
down to `d_model//downsample` once via a new `in_proj`, then runs every
internal chain op (embeds/attn/conv/ssm-state/head) at that smaller width
instead of full `d_model`, for all three `bit_head_class` variants
uniformly. Verified fixed/loop-consistent at every downsample factor.
**Helps train cost** (roughly halves-to-thirds the slowdown-vs-linear
going 1x→4x, e.g. attn 25x→16x, conv 26x→12x, ssm 32x→14x at dq=32; params
drop sharply too, e.g. attn dq=32: 5345→833). **Barely moves decode
cost** and is sometimes non-monotonic there (attn dq=32: 728x→583x→653x
across 1x/2x/4x) — decode's dominant cost is Python-loop/dispatch
overhead from dq sequential calls, not per-call matmul width, so
shrinking the matmuls doesn't touch the actual bottleneck. Net:
`bit_inner_downsample` is a solid free-ish training-time lever, not a fix
for generation-time cost — `ssm` remains the cheapest decode option
regardless of downsample.
