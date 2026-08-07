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

All 8000-step runs unless noted, batch_size=16, `datasets/enwik8_1M.gz`.
**mean it/s recomputed this session** for every row here and in the
`qcute_refine_v2` table below, all via the SAME consistent method —
`last_logged_step / elapsed_s_at_that_row` from each run's own
`run.jsonl` (includes periodic-eval overhead, not train-only throughput;
this is why some numbers here differ from earlier informal in-session
estimates — those used inconsistent/unspecified methods, these don't).
Best val_bpb/step computed from each log's final (non-stale) segment.

| run | context | best val_bpb | @ step | mean it/s | wall time |
|---|---|---|---|---|---|
| `bytelm_xs_mtp4_ctx1024` | 1024 bytes | **2.365** | 1700 | 0.904 | 2:27:31 (8000 steps) |
| `bpelm_8192` | 256 tok (~845 byte-equiv) | **2.350** | 800 | 3.878 | 0:34:25 (8000 steps) |
| `bpelm_32768` | 256 tok (~973 byte-equiv) | **2.134** | 500 | 1.971 | 1:07:41 (8000 steps) |
| `bytelm_xs3_ctx1024` (3-layer, this session) | 1024 bytes | **2.408** | 2100 | 1.563 | 0:42:39 (4000 steps) |
| ~~`qcute_refine_v2_byte4_code256_simple` ("v1")~~ | 1024 bytes | ~~**2.485**~~ | 5600 | 2.42 | 1:00:53 (8000 steps) |

**This config/run was later DELETED this session** (see the "Full
qcute_refine_v2 ablation-family comparison" section's own CORRECTION
note below for why — its actual cross_attn_rope status at training time
was ambiguous, not the confirmed value this row implied). Left struck
through, not removed, so this table's own history stays intact; treat
`qcute_refine_rope` (in the later table) as its replacement reference
point instead.

`qcute_refine_v2`'s "v1" run: worse best-bpb than either bpelm variant,
better than plain `bytelm`, but at ~2.1x `bytelm`'s throughput and ~4.3M
fewer params (2.706M vs. bytelm's 3.412M) — see the FLOPs/param
comparisons earlier in this session's conversation log for the fuller
picture (not yet transcribed into this file). Every baseline here
overfits well before 8000 steps (see the step-budget finding below) —
these best-bpb numbers, not the final-step ones, are the actual
comparison target.

## Params/FLOPs comparison table (this session)

Two new closer-param-matched baselines added this session:
`configs/bytelm_xs3_ctx1024.py` (3-layer variant of `bytelm_xs_mtp4_ctx1024`
— same d_model=256/n_heads=4/mtp_heads=4/context=1024, only n_layers 4->3;
required adding a `--n_layers` override flag to `qcute/bytelm.py`, which
previously only exposed `--context`/`--mtp_heads` as preset overrides) and
`configs/bpelm_16384_xs3.py` (3-layer, vocab=16384, same d_model=256/
context=256 pattern as `bpelm_32768.py` — `--n_layers` was already a
supported override on `qcute/bpelm.py`). Both `steps=4000` (this session's
step-budget finding). Reason: want baselines closer in params/depth to
the `qcute_refine` lineage's own 2-3 level towers than the original
4-layer/8000-step bytelm/bpelm baselines were.

FLOPs = single forward pass, batch_size=1, real op count via
`torch.utils.flop_counter.FlopCounterMode` (same methodology as
`scripts/bench_forward.py`), CPU, using each config's own context
length. Sorted by params:

| run | params | flops/fwd (batch=1) |
|---|---|---|
| `qcute_refine_v1` | 1.244M | 2.916G |
| `bytelm_xs3_ctx1024` | 2.625M | 5.369G |
| `qcute_refine_pass_through` | 2.642M | 3.695G |
| `qcute_refine_rope` | 2.706M | 3.862G |
| `qcute_refine_v2_byte4_code256_identity` | 2.706M | 3.862G |
| `qcute_refine_rope_3level_curriculum` | 3.414M | 4.330G |
| `qcute_refine_decoder_trunk` | 4.424M | 5.878G |
| `bpelm_16384_xs3` | 6.561M | 3.355G |

`bytelm_xs3_ctx1024` (2.625M) lands almost exactly on top of most
2-level `qcute_refine_v2` configs (2.6-2.7M) — a much closer param match
than the original 4-layer bytelm baseline (3.412M) ever was, at roughly
similar FLOPs too (5.4G vs. 3.7-3.9G — bytelm still costs more per
forward pass at matched params, consistent with its dense full-context
attention every layer vs. `qcute_refine`'s windowed/hierarchical
attention). `bpelm_16384_xs3` (6.561M) is far larger than everything
else here — its 16384-vocab embed+unembed tables dominate — so it's not
a fair params-matched comparison to anything in this table despite its
FLOPs/fwd being the lowest overall (context=256 tokens is a much shorter
sequence than the byte-level runs' context=1024).

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

**[DELETED later this session — config, logs, checkpoints all removed.**
Section kept as historical record of what was found while it existed;
see the "Full qcute_refine_v2 ablation-family comparison" section's own
CORRECTION note for why it was deleted (ambiguous cross_attn_rope status
at actual training time). Do not use `qcute_refine_v2_byte4_code256_
simple` as a live reference point anywhere else in this file — use
`qcute_refine_rope` instead.]**

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

## torch.compile: root-caused, fixed properly, still not a net win on MPS at this scale (this session)

`--compile` (new CLI flag on `qcute/qcute_refine_v2.py`, default `False`,
no-op unless passed): went through two wrong fixes before the real one.

**Attempt 1 (wrong)**: `model = torch.compile(model)` on the whole
`RefineLM`. Measured net SLOWER than eager (200 real steps,
`configs/qcute_refine_rope.py`, MPS: eager steady ~2.99 it/s, compiled
never caught up, ~2.5 it/s and still climbing). Root cause found via
`TORCH_LOGS=recompiles`: `RefineLM.forward` took the raw training-step
int `step` and branched on its exact value inside `n_active_levels()`
(gating `layer_warmup_steps`) — dynamo guards on that exact int and
recompiled almost every single step, even for configs that never set
`layer_warmup_steps` at all, since `step` still reached the compiled
function as a live-changing value.

**Attempt 2 (workaround, since removed)**: assert `layer_warmup_steps`
empty, pass `step=None` into the model whenever compiling. Fixed the
recompile problem but disabled curriculum+compile entirely.

**Attempt 3 (the real fix, what's in the code now)**: `RefineLM.forward`
now takes `n_active: int | None` directly instead of raw `step`.
`train()`/`eval_model()` compute `n_active = model.n_active_levels(step)`
themselves, in plain eager Python, and pass that in — `step` itself never
reaches the (possibly compiled) model call. Dynamo now guards on
`n_active`, which only takes a handful of distinct values across an
entire run (one per curriculum stage transition) instead of one per
step. Verified with a real 3-stage curriculum (`layer_warmup_steps=
(5,5)`, 20 steps crossing both transitions): every recompile event tied
to a genuinely new tensor shape appearing for the first time (a new
level activating really is a different compute graph), clustered right
at the two transitions, zero guard failures mention `step` anywhere.
Correctness verified throughout (matched-seed eager vs compiled,
multiple architectures/curricula): loss diffs ~1e-6, grad diffs ~1e-9,
the normal float32 op-reordering noise floor, no regression at any
point in this process.

**Real MPS speed result** (200 real steps, `qcute_refine_rope`'s
non-curriculum config, so this isn't confounded by the old recompile
bug): eager ~69s/199 steps (2.88 it/s), compiled (fixed) ~80s/199 steps
(2.49 it/s) — **still ~13-15% SLOWER than eager**, even with the
recompile bug genuinely gone. Likely cause: `d_model=256` means small
matmuls, and MPS/Inductor's kernel-launch overhead + less mature fusion
coverage (vs. CUDA) probably dominates over any compile-time fusion win
at this scale. **Conclusion: `--compile` is correct and available, but
not worth using on this hardware/model-scale combination — stays off by
default.**

As a side effect of this work, `DecoderLevel`'s cross-attention KV mask
(`n_complete = (t_idx+1)//K`, `visible = b_idx < n_complete`) was
re-verified correct — a "jagged staircase" pattern, empirically confirmed
for K=4: visible block count steps +1 every K raw positions, but the
step boundary lands at `t=(b+1)*K-1` (not a K-multiple) — block b's code
depends on `EncoderLevel`'s hidden state at that block's LAST position,
so it isn't causally available until that byte has been seen; the
formula gets this exactly right (no off-by-one, no label leakage).
Documented directly in the mask's own code comment in
`qcute_refine_v2.py`.

## New closer-matched baselines + full ablation-family comparison table (this session)

**`configs/bytelm_xs3_ctx1024.py`** (3-layer, d_model=256/n_heads=4/
mtp_heads=4/context=1024 — required adding a new `--n_layers` CLI
override to `qcute/bytelm.py`, which previously only exposed
`--context`/`--mtp_heads`) and **`configs/bpelm_16384_xs3.py`** (3-layer,
vocab=16384, d_model=256/context=256) added as closer-param-matched
baselines to the `qcute_refine_v2` 2-level configs than the original
4-layer/8000-step `bytelm`/`bpelm` baselines were. Full params/FLOPs
grid search (strict power-of-2, `n_layers=3` fixed) done for both —
see each module's own docstring (`qcute/bytelm.py`, `qcute/bpelm.py`,
"Session notes" sections) for the complete bytes/token, params, and
FLOPs tables plus the params-matched vs. FLOPs-matched config picks
against every `qcute_refine_v2` ablation target. Three new fair-comparison
bpelm configs came out of that search: `configs/bpelm_4096_paramsmatch.py`
(params-matched to `rope_3level_curriculum`, near-exact),
`configs/bpelm_16384_ctx448_flopsmatch.py` (FLOPs-matched to
`decoder_trunk`, near-exact), `configs/bpelm_8192_ctx448_flopsmatch_rope.py`
(FLOPs-matched to `rope`/`identity`) — queued to run after
`bytelm_xs3_ctx1024`.

**Full `qcute_refine_v2` ablation-family comparison** (params/flops =
single forward pass batch=1, `FlopCounterMode`; best val_bpb/step and min
train bpb from each run's own `run.jsonl`; mean it/s = same recomputed
method as the baseline table above, `last_logged_step / elapsed_s`,
includes eval overhead):

| run | params | flops/fwd | best val_bpb | @ step | min train bpb | mean it/s |
|---|---|---|---|---|---|---|
| `qcute_refine_v1` (module, stopped early @1050/8000) | 1.249M | 2.916G | 4.2202 | 1000 | 3.9936 | 0.165 |
| `qcute_refine_rope` | 2.706M | 3.862G | 2.6310 | 3600 | 1.8923 | 0.656 |
| `qcute_refine_pass_through` | 2.642M | 3.695G | 2.5575 | 3300 | 1.9614 | 2.149 |
| `qcute_refine_decoder_trunk` | 4.424M | 5.878G | 2.5793 | 2800 | 1.8086 | 1.407 |
| `qcute_refine_v2_byte4_code256_identity` | 2.706M | 3.862G | 2.5868 | 3800 | 1.8957 | 2.545 |
| `qcute_refine_rope_3level_curriculum` | 3.414M | 4.330G | 2.6463 | 3300 | 1.9569 | 0.484 |

Notable it/s spread despite `rope`/`pass_through`/`identity` sharing near-
identical params (2.6-2.7M): 0.656 vs. 2.149 vs. 2.545 it/s — a ~3.9x
range. `identity` (no BSQ quantization) and `pass_through` (no encoder-
hidden-state reuse on either decoder side) are both cheaper per-step than
`rope`'s full BSQ+reuse path despite near-identical FLOPs/fwd and params,
suggesting real per-step overhead outside the counted forward-pass FLOPs
(quantization op, extra backward-graph complexity from hidden-state
reuse) — worth a closer look if throughput matters more than architecture
purity for future runs. `decoder_trunk` (private trunk copies, most
params/flops) and `rope_3level_curriculum` (3 levels, more sequential
work) are slowest, as expected from their own higher params/flops.
`qcute_refine_v1` is by far the slowest (0.165 it/s) — its own module
uses BitPredictHead chain-mode NTP heads throughout (unlike v2's
`code_head_mode="independent"` runs), consistent with this session's own
BitPredictHead speed findings (see the dedicated section above).

**CORRECTION (superseding this section's original text): `configs/
qcute_refine_v2_byte4_code256_simple.py` and its results (logs/
checkpoints) were DELETED from this table and from the repo.** The
original claim here — that `qcute_refine_rope` was "functionally a
duplicate" of `simple` because both used `cross_attn_rope=True` — was
wrong. It compared `simple`'s config text against `Config`'s CURRENT
default, but `simple`'s actual historical run happened BEFORE the
`cross_attn_rope` feature (and its default) existed in the codebase at
all — at the time it trained, cross-attention had no RoPE option, period,
so that run's cross-attention was genuinely position-blind (i.e. closer
to a `cross_attn_rope=False` run than a `True` one), not the confirmed
`True` run this table previously implied. Rather than keep a config/
result pair whose actual historical rope status is ambiguous, it was
deleted outright. The real rope-vs-no-rope ablation is now
`configs/qcute_refine_rope.py` (confirmed `cross_attn_rope=True`, in the
table above) vs. `configs/qcute_refine_no_rope.py` (cloned directly from
`rope.py`, only `cross_attn_rope=False` changed, same 4000-step budget)
— queued, not yet run as of this note.

Other reads from the table: `qcute_refine_v1` stopped at step 1050 of a
planned 8000 (superseded early when work moved to v2) — not a fair
endpoint comparison, shown for completeness only. Among the remaining
4000-step same-family runs (`rope`/`pass_through`/`decoder_trunk`/
`identity`/`rope_3level_curriculum`), results cluster tightly
(2.5575-2.6463) — no ablation here produced a dramatic swing;
`pass_through` (cheapest architecture, zero encoder-hidden-state reuse on
either side of the decoder) actually edges out the others slightly
despite being the most stripped-down. `decoder_trunk` is the most
expensive (4.424M params, 5.878G flops — private trunk copies aren't
free) without a proportionate quality win.

## `configs/qcute_refine_tiny_byte_window.py` — queued, forcing cross-attention to matter

Clone of `qcute_refine_rope.py` with `attn_window` changed `(256, 64)` ->
`(8, -1)`: level 0 (byte encoder) window shrunk to an extremely tiny 8
raw bytes (self-attention alone can see almost nothing), level 1 (code
encoder) set to dense/full attention over its own 256 code positions (its
own effective receptive field now spans the WHOLE 1024-byte context).
Rationale: if the KV-contribution probes' mixed/inconclusive findings on
`simple`/`rope`-family runs were partly because level 0 already had
plenty of local context (its own `attn_window=256` already covers a full
256-byte lookback) to lean on instead of the cross-attention, this
config removes that alternative almost entirely — level 0 is nearly a
bag-of-8-bytes model on its own, so genuine cross-attention KV
contribution (if any) should show up starkly in a
`probe_decoder_kv_contribution.py` re-run against its checkpoint. Smoke-
tested (3 CPU steps, no shape/divisibility errors, no dense-fallback
warnings) before queueing. Queued to run after the three new bpelm
fair-comparison configs above.
