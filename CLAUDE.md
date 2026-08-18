# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                         # install/update env from pyproject.toml + uv.lock
uv run python scripts/prepare_data.py           # download/cut datasets/enwik8{,_1M}.gz
uv run python scripts/train_bpe.py --data datasets/enwik8_1M.gz   # BPE tokenizer for qcute.bpelm
uv run python -m qcute.bytelm --preset sd       # byte-level baseline LM (Phase 0), reports BPB
uv run python -m qcute.archive.qcutelm          # end-to-end tokenizer + latent LM (FSQ/BSQ) — ARCHIVED,
                                                 # superseded by qcute_refine (see Architecture below); still
                                                 # importable/runnable via its archive path for historical reference
uv run python -m qcute.bpelm --sp_model datasets/bpe_enwik8_1M_8192.model   # BPE baseline
uv run python -m qcute.bytelm --config configs/bytelm_xs_mtp4_ctx1024.py   # named, reproducible run — the
                                                 # standard byte-level baseline as of this session (context=1024,
                                                 # matching qcute_refine's own context_len); the older
                                                 # configs/bytelm_xs_mtp4.py (context=256) is superseded, kept only
                                                 # for historical reproducibility, not a comparison target anymore
uv run python scripts/plot_run.py logs/<run_name>   # train/val bpb PNG from a run's run.jsonl
uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_3.py
uv run python -m qcute.qcute_v5_stack --config configs/qcute_v5_stack_3.py
                                                 # the two DEFAULT v5 modules (see Architecture below):
                                                 # qcute_v5_concat.py and qcute_v5_stack.py — both
                                                 # promoted from their `_skip` forks (buffer-pruning:
                                                 # once a block's code exists, its raw bytes are
                                                 # dropped from the decode buffer), which forked from
                                                 # `_fixblock` (qfb boundary-query mechanism removed,
                                                 # replaced with decode_bos removal + block-0 target
                                                 # exclusion) — each self-contained, run directly. The
                                                 # prior defaults (qfb-based, pre-fixblock/pre-skip) are
                                                 # ARCHIVED as qcute/archive3/qcute_v5_bos.py and
                                                 # qcute/archive3/qcute_v5_concat_bos.py, kept for
                                                 # historical reference only. Intermediate forks kept as
                                                 # comparison references: qcute_v5_fixblock.py,
                                                 # qcute_v5_concat_fixblock.py, qcute_v5_slow.py,
                                                 # qcute_v5_concat_slow.py, and the weight-sharing
                                                 # variant qcute_v5_ws_slow.py
```

`qcute.bytelm`, `qcute.bpelm`, `qcute.qcute_v5_concat`, and
`qcute.qcute_v5_stack` all read `--help` for their full flag
list; all support `--config path.py` (see `configs/` — every config file
has its own module docstring explaining what it's testing and its exact
`uv run` invocation, copy-pasteable directly from the file), `--run_name`
(else derived from `--config`/preset — logs and checkpoints both key off
it: `logs/<run_name>/`, `checkpoints/<run_name>/`), and `--eval_only
--checkpoint_path ...`; `qcute.bytelm` additionally supports
`--qual_gen_bytes` for qualitative generation. Tiny-corpus-scale defaults
(`xs` preset) target ~4 bytes/timestep — see `qcute/bytelm.py`'s
`PRESETS` comment for why. No test suite, linter, or CI config exists yet.

**Only ever run one training job at a time.** All four modules train on
MPS; two concurrent training processes contend for the same GPU and both
slow down (observed directly: a second run caused an already-progressing
job to stall with zero throughput). Kill or wait out the current run
before launching another — never launch a second training process while
one is still active.

**When launching a training run in the background, never redirect its
stdout/stderr to `/dev/null`** — use a file instead (e.g. a scratchpad
path, or `/tmp/<pid>.log` renamed once the PID is known post-launch),
since `/dev/null` silently swallows uncaught-exception tracebacks and
anything not routed through `Logger`, making crashes invisible. Do NOT
pipe through `tr '\r' '\n'` to make tqdm's `\r`-updates readable —
confirmed directly that `tr` itself full-block-buffers its own stdout
when writing to a non-tty file, so a `tail -f` on the piped-through file
sits empty for seconds/minutes at a time regardless of how eagerly the
Python process flushes its side of the pipe; it's not a live view, just a
deferred dump. Redirect stdout/stderr straight to a file instead (plain
`... > /tmp/foo.log 2>&1 &`, no pipe) — the tqdm line will look like one
long `\r`-joined blob when catted, but `tail -f` still shows new bytes
arriving in real time, which is the actual goal. Use `pgrep -f "python3
-m qcute.<module>"` to find the training process's PID (e.g. to kill it;
`$!` after a background launch gives the wrapper/shell PID, not
necessarily Python's). **After launching, give the user the PID and two
`tail -f` commands**: one on that raw stdout/stderr file, and one on
`logs/<run_name>/run.log` (the structured log `Logger` writes to at
`--log_every`/`--eval_every` intervals, genuinely real-time since
`Logger` opens and flushes that file directly, no pipe involved) — so
they can watch it live themselves rather than relying on being told the
outcome later. Long runs have shown unpredictable throughput (observed: a
nominal ~30-minute budget taking 2.5-3.5 hours instead) — watch actual
elapsed time/step rate early on rather than assuming a run will finish on
schedule.

## Architecture

`qcute/bytelm.py` and `qcute/bpelm.py` are self-contained baseline
modules — none import each other, deliberately not factored further yet.
Full details: [docs/architecture.md](docs/architecture.md) (also covers
`qcutelm.py`, now archived — see below).

**The entire `qcute_refine` lineage (`v1.py` through `v4_5_1.py`, plus
its `qcute_refine.py` always-latest alias) is now archived** under
`qcute/archive2/` (configs under `configs/archive2/`; its docs —
`qcute_refine_math.md`, `kv_contribution.md`, `bpe_like_boundaries.md`,
`bitpredict_heads.md`, `torch_compile.md` — under `docs/archive2/`).
**`qcute/qcute_v5_concat.py` and `qcute/qcute_v5_stack.py` (forked from
`qcute_refine_v4_4_1.py`/`qcute_refine_v4_5_1.py` respectively, dropping
the `refine_` prefix and the version-suffix/alias convention entirely)
were the original two standalone prototypes** — each self-contained, run
directly (no promotion step, no alias file). Both have since been
superseded as the *default* module by an efficient-attention/KV-cache-
friendly rewrite, while being kept themselves as O(L^2) dense references:

- **`qcute_v5_stack.py`** → superseded by **`qcute/qcute_v5.py`**
  (renamed from an intermediate `qcute_v5_stack_eff.py` fork once it
  became the default): genuinely sub-quadratic windowed/banded attention
  (chunked `selfcode_decode`/`cross_attn_stage`, no dense `O(L^2)` mask)
  and a FIFO-windowed `generate_kv_cache` (truncate to `context_len` each
  step, cheap and structurally cache-friendly — not yet a true per-layer
  K/V tensor cache). Verified via `scripts/test_v5.py` against
  `qcute_v5_stack.py` (bit-identical forward/loss across Ks/window
  shapes) and `check_gen_consistency`/`validate_generation`.
  **Superseded again**: weight-sharing logic (`share_level_weights`,
  `decode_separate_stage0`) pruned to its always-`False` default (the old
  variant kept as `qcute_v5_ws_slow.py`), then a "query first byte" (qfb)
  fix folded into `cross_attn_stage` itself, removing `selfcode_decode`
  entirely — see "qfb boundary-query fix" below. The pre-qfb file is kept
  as `qcute_v5_slow.py`.
- **`qcute_v5_concat.py`** → rewritten in place (the old dense
  implementation is now **`qcute/qcute_v5_concat_slow.py`**): every
  track's codes are placed at their true chronological time position —
  merged with the byte stream into ONE physically time-ordered buffer
  per level's decode — instead of the old "prepend" scheme (all track
  prefixes grouped at the buffer front, corrected via a separate
  `true_pos` array + `argsort`). Buffer order now IS time order, so
  causal masking is a plain buffer-index comparison (no same-position
  exclusion mask term needed — a tied code always sorts after the byte
  that produced it, automatically invisible to it) and windowed/banded
  attention slices CONTIGUOUS buffer ranges with no runtime sort — the
  index/address construction (`LevelLM._merged_layout`) depends only on
  shape, never data, so it's built once per signature and cached, same
  cost whether called every training step or every fixed-size
  `generate_kv_cache` FIFO-window step. Single- and multi-track decode
  are now one mechanism (no more separate selfcode/dense/banded code
  paths). An intermediate fork, `qcute_v5_concat_eff.py` (argsort-based:
  kept the old "prepend" layout, cached the sort's structural output
  rather than eliminating it), exists between `qcute_v5_concat_slow.py`
  and the current `qcute_v5_concat.py` but was never itself promoted/
  renamed. Verified via `scripts/test_v5_concat.py` (an independent
  from-scratch dense reference, dense-vs-chunked internal consistency,
  `check_gen_consistency`, `validate_generation` — not compared against
  `qcute_v5_concat_slow.py` directly, since window semantics changed by
  design, see the module's own docstring).

**`qcute_v5_concat.py` promoted again (2026-08-17)**, from a
`qcute_v5_concat_modes.py` fork adding per-conditioning-depth multi-mode
decode loss (`Config.multi_mode_impl: "off"|"multipass"|"single_pass"`,
mirroring `qcute_v5_stack.py`'s free `decode_stage_extra_losses`
byproduct, which flat self-attention decode has no natural equivalent
of): `"off"` (default) is bit-exact with the pre-fork behavior — that
version is kept as `qcute/qcute_v5_concat_no_modes.py`, a strict no-op
reference. `"multipass"` is a naive T-calls-per-level reference;
`"single_pass"` batches every mode's own independent merged buffer
(block-diagonal masked, zero cross-mode attention) into one shared pass
— exact vs. `"multipass"`, supports both the dense and the chunked/
banded (SWA) attention path (`_merged_layout` extended with
`forced_sc`/`forced_n_chunks`/`forced_n_prev_chunks`, safe to force a
chunk grid larger than a segment's own natural need). Verified via
`scripts/test_v5_concat_modes.py` across `Ks=(1,)`/`(4,1)`/`(2,2,1)`,
dense and chunked.

Both `qcute_v5.py` and `qcute_v5_concat.py` also had several `Config`
flags hardcoded away (`decode_code_ste` always `True`,
`cross_track_source` always `"decode"`, `decode_self_only_aux` and its
curriculum loss removed entirely — decode now has exactly one NTP loss
term) and their `quant_type` dispatch (`"softmax"` vs `"bsq"`) unified
into a `QuantScheme`/`SoftmaxQuant`/`BSQQuant` strategy-class pair (7
uniform methods: `init_modules`/`quantize`/`to_ids`/`embed_for_decode`/
`ntp_loss_acc`/`embed_input`/`sample_next`) — `make_quant(cfg)` is the
only remaining `quant_type` branch in either file, everywhere else
dispatches through `self.quant.<method>()`.

**`QuantScheme` gains a third implementation, `FSQQuant` (2026-08-17)**,
in both `qcute_v5_stack.py` and `qcute_v5_concat.py`: `Config.quant_type="fsq"`
(finite scalar quantization, Mentzer et al. 2023, ported from archived
`qcute/archive/qcutelm.py`), `Config.fsq_dq`/`fsq_levels` (defaults 6/8).
`Config.fsq_bound` picks the per-dim squashing nonlinearity applied
before rounding — `"sigmoid"` (default, iFSQ, matches archived
`qcutelm_vlt6.py`'s own default) or `"tanh"` (original FSQ) — one quant
type plus a sub-flag rather than two separate `quant_type` strings,
since embedding/loss/sampling are identical either way. `BSQQuant` gains
`Config.bsq_lfq` (default `False`, unchanged behavior): skips the
L2-normalize-before-sign step, hypercube corners instead of BSQ's
hypersphere. Also fixed a real pre-existing bug found while touching
this code: `BSQQuant.sample_next` referenced a nonexistent
`self.use_bernoulli_sample` (should've been `self.mode`) — would have
crashed on the first BSQ generation call in either file.

**BSQ entropy regularization (2026-08-17)**, both files: `bsq_entropy_reg`
(Yu et al. 2023 §3.2 MAGVIT-v2 / BSQ 2024 closed-form, ported from
archived `qcutelm.py`) via a new `QuantScheme.entropy_reg(pre_q)` hook
(default `None`; only `BSQQuant` overrides it), weighted by
`Config.entropy_reg_weight`/`--entropy_reg_weight` (default `0.0`, off).
Threading the term from each level's raw `pre_q` up to the loss required
real return-tuple arity changes (not just new optional params): in
`qcute_v5_concat.py`, `LevelLM.forward` 8-tuple→9-tuple and `RefineLM._run`
11-tuple→12-tuple; in `qcute_v5_stack.py`, `LevelLM._extract_code` gained a
second return value, `LevelLM.encode` 4-tuple→5-tuple, and `RefineLM._run`
12-tuple→13-tuple — every call site of each (generation/diagnostic
functions included, ~13 per file) updated to match. Deliberately did NOT
reuse `qcute_v5_concat.py`'s existing always-`None` 6th `LevelLM.forward`
slot even though it looked unused — `check_gen_consistency` branches on
it via `is not None`, so overloading it would have silently mis-routed
that check whenever entropy_reg was non-None; a new trailing tuple
element was added instead.

**`qcute_v5.py`'s "query first byte" (qfb) fix**: `cross_attn_stage`'s
strict causal mask (`code_pos < query_pos`) means the row that would
predict a block's FIRST element from that block's own just-completed
code is the same row the code was derived from — excluded by
construction, so a code only ever conditions predictions from a block's
SECOND element onward. Fixed with one extra, unconditionally-patched
cross-attention query per block (`LevelLM.decode_boundary_query`, a
shared fixed vector, sidesteps the chicken-and-egg problem since nothing
computed it FROM the code it attends to) — applied identically to every
track at every level (self or coarser), replacing `selfcode_decode`
entirely; `RefineLM._run`'s decode loop is now one uniform per-level loop
over every track, no track-index or level special-casing. Two real bugs
surfaced and fixed while building this (self-referential code extraction;
a length-dependent internal patch that broke KV-cache append-only
semantics — every block's boundary row is now patched unconditionally,
never conditioned on "is this the last block visible in this call", so a
row's content depends only on its own causal past, never on how much
sequence follows it within the current call). The old level1-only
diagnostics (`generate_level1_codes`, `generate_level1_codes_via_decode`,
`level1_ground_truth_codes`) are generalized to `generate_level_codes`
etc., parameterized by `level`. Full narrative, the bug hunt, and a
concrete `n_levels=1/2/3` byte-level walkthrough:
[docs/status.md](docs/status.md).

Configs: `configs/qcute_v5_concat_*.py`, `configs/overfit/qcute_v5_concat_*.py`
(now running against `qcute_v5_concat_slow.py`, repointed at rename time),
`configs/overfit_concat_eff/`, `configs/overfit_stack_eff/`,
`configs/qcute_v5_stack_*.py`.

Archived lineage summary, oldest to newest:
- `qcute_refine_v1.py`/`v2.py`/`v3.py`: pre-v4 history — BSQ code
  hand-off with a joint-chain-MTP detokenizer, then a cross-attending
  `DecoderLevel`, then EncoderLevel fusion.
- `qcute_refine_v4.py`: removes `DecoderLevel` entirely — it turned out
  to do the literal same job as fusion (predict a level's own next
  token, optionally conditioned on the coarser code) and never
  contributed to `byte_loss` even in v3. `EncoderLevel` renamed to
  `LevelLM` (the class now does both the "encode" job — PASS 1, produce
  the code — and the "decode" job — PASS 2, fused/conditioned
  prediction — so "Encoder" alone was misleading). Generation
  (`generate_no_cache`/`generate_kv_cache`) is fusion-aware — v3's own
  generation functions were copied unchanged from v2 and never touched
  cross-attention at all, a real train/inference mismatch v4 fixes.
  `qcute.bytelm`/`qcute.bpelm` also gained matching `generate_kv_cache`/
  `validate_generation` this session, same pattern.
- `qcute_refine_v4_4.py`/`qcute_refine_v4_5.py`: v4.4 adds
  packed-sequence multi-track cumulative decode (self + every coarser
  level's code, `decode_pack_mode`); v4.5 replaces that with explicit
  staged cross-attention through the same shared weights (no packed
  sequences). Both support `Config.share_level_weights` (default `True`,
  original behavior unchanged) — `False` gives every level (v4.4) or
  every level's own encode LM plus one independent LM per decode
  cross-attention track (v4.5) fully independent weights, coupled only
  through the bare integer code id crossing between them.
- `qcute_refine_v4_4_1.py`/`qcute_refine_v4_5_1.py`: fix the self
  track's decode conditioning to be genuine LM continuation (`code_b`
  conditions block `b+1`, never its own block `b` — the original
  v4.4/v4.5 mechanism was an accidental autoencoder for the self track
  specifically). Math:
  [docs/qcute_refine_v4_4_1_v4_5_1_math.md](docs/qcute_refine_v4_4_1_v4_5_1_math.md)
  (this doc itself is not archived — its "self-code LM continuation"
  mechanism carries over unchanged into `qcute_v5_concat.py`/
  `qcute_v5_stack.py`, see below).

`qcute_v5_concat.py` (from v4.4.1's packed self-attention decode) and
`qcute_v5_stack.py` (from v4.5.1's staged cross-attention decode) add
`Config.quant_type: "softmax"` (default, unchanged categorical code) or
`"bsq"` (binary spherical quantization, `Config.bsq_bits`-wide sign
code, straight-through) as an alternative to the categorical code
representation. Full detail on all of the above, including an ongoing
slow-convergence investigation (a moving-target/cascade effect for
n_levels=2 configs, compounded in the "stack"/v4.5-style decode by a
much deeper gradient path back to the code producer — decode is now
`n_layers * (1 + n_tracks)` deep sequentially, vs. the "concat"/v4.4-style
decode's flat `n_layers`), and the `decode_code_ste` findings (predates
`qcute_v5.py`/current `qcute_v5_concat.py` hardcoding `decode_code_ste`
to always `True` — see above):
[docs/status.md](docs/status.md) (reset partway through the v5 work —
full pre-reset history at
[docs/archive2/status.md](docs/archive2/status.md)).

**Standing methodology**: use a small (`n_bytes=10000`) slice of the
corpus with a short step budget as the standard fast-iteration testbed
for v5 architecture changes — see `configs/*_overfit10k_*.py` — until a config can actually fast-overfit
that slice to a train bpb comparable to `qcute.bytelm`'s own parity
numbers on the same slice (`n_layers=1`: 0.0212 train bpb at step 1000,
~19.7 it/s; `n_layers=2`: 0.0072, ~11.4 it/s — see
`configs/bytelm_overfit10k_*.py`). Full-scale (~900k-byte) runs and
generation-quality comparisons are not trustworthy until that parity
bar is cleared — a config that hasn't even memorized a 10k-byte slice
yet tells you little about its behavior at scale.

**Ks regression grid, simplest to hardest** (for config-writing later): ranked by
`product(Ks)` (compression ratio / minimum warm-up context) first, `n_levels` second,
`max(Ks)` third.

| # | Ks | levels | product(Ks) | max K |
|---|---|---|---|---|
| 1 | `(1,)` | 1 | 1 | 1 |
| 2 | `(1,1)` | 2 | 1 | 1 |
| 3 | `(1,1,1)` | 3 | 1 | 1 |
| 4 | `(2,1)` | 2 | 2 | 2 |
| 5 | `(2,1,1)` | 3 | 2 | 2 |
| 6 | `(2,2)` | 2 | 4 | 2 |
| 7 | `(4,1)` | 2 | 4 | 4 |
| 8 | `(2,2,1)` | 3 | 4 | 2 |
| 9 | `(4,2)` | 2 | 8 | 4 |
| 10 | `(2,2,2)` | 3 | 8 | 2 |
| 11 | `(4,2,1)` | 3 | 8 | 4 |
| 12 | `(4,4,2)` | 3 | 32 | 4 |

Ranks generation/architecture-correctness difficulty (raggedness, warm-up depth), not
training/learnability difficulty — a high-product config may be easier to overfit10k
(fewer effective tokens) despite being harder to verify `check_gen_consistency` on.

Diagnostic: `scripts/probe_decoder_kv_contribution.py` (gradient/
ablation/attention-mass analysis of how much cross-attention KV actually
contributes vs. is ignored — run against a saved v2/v3 checkpoint, now
under `qcute.archive2.qcute_refine_v2`; v4-and-later have no
`DecoderLevel` for this script to target). Historical narrative on the
pre-v4 lineage (v2's `DecoderLevel` KV contribution, boundary-adaptivity
brainstorm, `BitPredictHead` speed comparisons, `torch.compile` on v2):
[docs/archive2/kv_contribution.md](docs/archive2/kv_contribution.md),
[docs/archive2/bpe_like_boundaries.md](docs/archive2/bpe_like_boundaries.md),
[docs/archive2/bitpredict_heads.md](docs/archive2/bitpredict_heads.md),
[docs/archive2/torch_compile.md](docs/archive2/torch_compile.md).

Several older one-off scripts (`scripts/compare_ste_vs_detach.py`,
`probe_code_usage_entropy.py`, `probe_v4_fusion_contribution.py`,
`qual_gen_v4_2.py`, `qual_gen_v4_3_vs_baseline.py`,
`test_v4_4_banded_decode.py`, `test_v4_4_chunked_decode.py`,
`qual_gen_bytelm.py`) still `import` archived `qcute_refine_vN` modules
by their pre-archive top-level path (e.g. `from qcute.qcute_refine_v4_4
import ...` instead of `from qcute.archive2.qcute_refine_v4_4 import
...`) and will `ImportError` if run as-is — not part of the two
maintained diagnostics above, left unfixed since nothing currently
depends on them; fix their import path the same way if you need to
revive one.

Every earlier qcute-lineage fork — `qcutelm.py`, `qcutelm_vlt*.py`
(`vlt` through `vlt11`), `qcutelm_pyramid.py`, `qcutelm_mergetoken_v1.py`,
`qcute_bytepool.py` — is **archived** under `qcute/archive/` (configs
under `configs/archive/`; their own design docs —
`continuous_tokenizer_handover.md`, `fifo_v2.md`, `vlt12_math.tex` — under
`docs/archive/`), superseded by the `qcute_refine` lineage. The
`qcute_refine` lineage itself (`v1.py` through `v4_5_1.py`, the pre-v5
history) is separately archived under `qcute/archive2/` — see above — a
different archive directory since it's a later, distinct lineage from
the `qcutelm.py` family. Kept for historical reference/reproducibility,
not part of active work. `qcute/bytelm.py` and `qcute/bpelm.py` are the
exception — still the active baseline comparison points, not archived.

Original design source-of-truth for the now-archived `qcutelm` lineage:
[docs/archive/continuous_tokenizer_handover.md](docs/archive/continuous_tokenizer_handover.md)
(historical — `qcute_refine`'s own design isn't specified by it). Chronological one-line summary
of every fork in both the `qcutelm` (`qcute/archive/`) and `qcute_refine` (`qcute/archive2/`)
lineages: [docs/archive/lineage_summary.md](docs/archive/lineage_summary.md).
Current progress: [docs/status.md](docs/status.md).

## Code style

Docstrings and comments must be extremely concise — assume code is self-descriptive; comments
never exceed 2 lines. Only the module-level (top-of-file) docstring is exempt from the length
limit, and should still be kept reasonably tight rather than accumulating restated history.

## Response format

Prefix every reply to the user with the current timestamp (run `date` for
the actual value — never guess it).

Keep chat replies terse. State results and next steps directly — no restating context the user
already has, no padding, no multi-paragraph recaps. Save detail for docs/status.md, not the chat.
