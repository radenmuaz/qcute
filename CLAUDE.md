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
uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks21_v256_pq1_overfit10k.py
                                                 # ACTIVE lineage: qcute_v1 (qcute/qcute_v1/) is where the
                                                 # latent-AR / parallel-block-local-decode rewrite (see below)
                                                 # is actually implemented -- forked from a verbatim copy of
                                                 # qcute_v5, diverging as of the BOS-interleaved-decode rewrite.
                                                 # Full design doc: docs/qcute_v1_plan.md.
uv run python -m qcute.v5_old.qcute_v5 --decoder_type stack --config configs/v5_stack_fsq/ks1_16x8.py
                                                 # ARCHIVED: qcute_v5 (formerly qcute/qcute_v5*.py, moved to
                                                 # qcute/v5_old/ once qcute_v1 became the active lineage) --
                                                 # still the leaderboard's source of truth for everything
                                                 # already run (docs/status.md), frozen, not receiving further
                                                 # architecture work. `--decoder_type concat` also still works.
```

`qcute.bytelm`, `qcute.bpelm`, `qcute.qcute_v1.qcute_v1`, and
`qcute.v5_old.qcute_v5` all read `--help` for their full flag
list; all support `--config path.py` (see `configs/` — every config file
has its own module docstring explaining what it's testing and its exact
`uv run` invocation, copy-pasteable directly from the file), `--run_name`
(else derived from `--config`/preset — logs and checkpoints both key off
it: `logs/<run_name>/`, `checkpoints/<run_name>/`), and `--eval_only
--checkpoint_path ...`; `qcute.bytelm` additionally supports
`--qual_gen_bytes` for qualitative generation. Tiny-corpus-scale defaults
(`xs` preset) target ~4 bytes/timestep — see `qcute/bytelm.py`'s
`PRESETS` comment for why. No test suite, linter, or CI config exists yet.
Existing `configs/v5_stack_*/`, `configs/v5_word/`, etc. docstrings still say
`qcute.qcute_v5`/`qcute.qcute_v5_wordlm` (pre-move path) in their copy-pasteable
`uv run` line — substitute `qcute.v5_old.qcute_v5`/`qcute.v5_old.qcute_v5_wordlm`
when actually running them; left as-is rather than bulk-edited, matching how
other archived lineages' configs keep their original invocation text.

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
necessarily Python's). **After launching, give the user the PID, two
`tail -f` commands, and (when the run is in a `tmux` session, e.g. on a
remote TPU node) a `tmux capture-pane` command**: one `tail -f` on that
raw stdout/stderr file, one on `logs/<run_name>/run.log` (the structured
log `Logger` writes to at `--log_every`/`--eval_every` intervals,
genuinely real-time since `Logger` opens and flushes that file directly,
no pipe involved), and `tmux capture-pane -t <session> -p -S -N` (peeks
at the session's recent output without attaching — swap `-N` for how many
lines back) alongside the `tmux attach -t <session>` command — so they
can watch it live themselves rather than relying on being told the
outcome later. Long runs have shown unpredictable throughput (observed: a
nominal ~30-minute budget taking 2.5-3.5 hours instead) — watch actual
elapsed time/step rate early on rather than assuming a run will finish on
schedule.

## TPU access

**Starting a fresh session asked to do a TPU run: read [docs/bytelm_tpu_setup.md](docs/bytelm_tpu_setup.md)
and [docs/tpu_direct_ssh.md](docs/tpu_direct_ssh.md) first, in full** — the first walks
scp → install → run → common failure modes (including two bugs already hit and fixed: the
torch/torch_xla version-pin ABI mismatch, and `xm.optimizer_step`'s `barrier=False` default
silently growing the XLA graph unboundedly across steps); the second is the direct-ssh
connection setup below, which is not optional — do it immediately, every time. Don't rediscover
any of this from scratch. [TPU.md](TPU.md) lists which queued resources exist (never create a
new one).

TPU VMs (see [TPU.md](TPU.md) for available queued resources) are reachable via
`gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone>`, but that
re-validates TPU state and re-preps the node on every call (several seconds of overhead each
time). **The very first thing to do on any fresh TPU node connection — right after confirming
it's `READY`, before install/scp/anything else — is set up the direct-ssh persistent multiplexed
connection** (one `gcloud ... ssh` call to propagate the key, then a `ControlMaster`/
`ControlPersist` session against the node's external IP) and use that for every subsequent
command on that node, not repeated `gcloud ... ssh` calls. **Every TPU listed in TPU.md is a
spot instance and can be preempted at any time with no warning** — if a node that was just
working suddenly can't be reached (hang, `Connection refused`, `No route to host`), check
`queued-resources describe ... state.state` for `PREEMPTED` *before* assuming a flaky connection,
retrying, or standing up a replacement node (don't — see above). Full setup and copy-pasteable
commands: [docs/tpu_direct_ssh.md](docs/tpu_direct_ssh.md).
**Never create/start a TPU yourself** — only use nodes already listed in TPU.md/already running.
**Never edit TPU.md itself** — it's the user's own list of queued-resource create commands, not a
session log; per-node details discovered while connecting (external IP, actual node name behind
a queued-resource name, accelerator type) go in a session's own working notes/commands, not
written back into TPU.md.
Full scp-to-running-training walkthrough (uv/torch_xla install, common failure modes, `qcute.
bytelm_tpu` smoke test): [docs/bytelm_tpu_setup.md](docs/bytelm_tpu_setup.md). **Any long-running
or user-monitorable remote command (installs, training) goes inside a `tmux` session on the TPU
VM**, not a bare blocking `gcloud ... ssh --command`, and the user gets the exact `tmux attach`
command back so they can watch it live themselves — see that doc's own tmux section for the
launch/attach/peek incantations. **For a multi-hour run, check in periodically (roughly hourly)
and pull back only `run.jsonl` (not `run.log` or checkpoints) to the matching local `logs/<run_name>/`
path to save egress** — see that doc's "Monitoring a multi-hour run" section for the exact
commands.

**`qcute.bytelm_tpu`'s `--use_flash_attention` needs a nightly torch/torch_xla build** (the
stable pin's `libtpu==0.0.21` is too old for the Pallas kernel) — full install steps, confirmed
working on a `v4-8` node: [docs/bytelm_tpu_setup.md](docs/bytelm_tpu_setup.md)'s "Optional:
nightly build" section. **`--multichip` (collective data-parallel across chips) WORKS on the
stable `torch==2.9.0`/`torch_xla==2.9.0` pin** (confirmed 2026-08-23 on a fresh `v4-8` node: 4
real worker processes, steadily climbing CPU time, a full run completed cleanly with
`world_size=4 global_batch=64` correctly reported after fixing a real bug — `world_size` must
come from `xr.world_size()`, not `xr.addressable_runtime_device_count()`, which returns 1 inside
an already-spawned worker) — the earlier "confirmed broken" hang was specific to the **nightly**
`torch_xla==2.10.0.dev0` build (only needed for `--use_flash_attention`), not this project's own
wiring, exactly as the 2026-08-22 static-code-review hypothesis predicted. `--multichip` +
`--use_flash_attention` together **is now confirmed to hang** (2026-08-23, fresh `v4-8` node,
nightly build) — CPU time flat across all 4 workers on a 5-step smoke test, the same signature as
the plain nightly-build hang below; standalone flash-attention works fine on the same node/build,
so it's specifically the combination that's broken. Not usable together on any build tried so far
(stable pin can't do flash-attention at all; nightly can do either alone, not both at once). See
docs/bytelm_tpu_setup.md's "Optional: multiple TPU chips on one host" section for the full
writeup. For single-chip-per-process (embarrassingly-parallel, e.g. a hparam sweep) instead of
collective multichip, launch independent processes one per chip via `TPU_VISIBLE_CHIPS=<i>`
(also verified working) — see that same doc section for the exact pattern. Check real addressable device
count first (`torch_xla.runtime.addressable_runtime_device_count()` / `ls /dev/accel*` /
`tpu-info`) — a slice's "-N" suffix (e.g. `v4-8`) counts TensorCores, not addressable devices.
**Multi-chip runs get one named `tmux` session per chip**, not backgrounded `&` jobs in a single
shell (those all die if that one shell's session ends) — `tmux ls` lists every session on the
node, `tmux capture-pane -t <session> -p -S -N` peeks at any one without attaching (swap
`<session>` per run), `-t muaz@<ip> "tmux attach -t <session>"` (note the `-t` before the ssh
target, for a real pty) attaches interactively.

**`qcute.bytelm_tpu`'s `zero_kv_sink` (default on) and `--use_flash_attention` don't mix well**:
a correctness fix exists (pad Q by one dummy row + pick `context=1024*k-1`) but even the correct
version costs ~25x steady-state throughput (10s/it vs. 0.4s/it) — confirmed not a `torch.compile`
or tensor-allocation artifact, and a from-scratch JAX reimplementation
(`qcute/bytelm_jax.py`) didn't rescue it either (a properly controlled sink-on/off test in JAX
showed the sink made no measurable difference there, ~0.33-0.35 it/s regardless — JAX's own
floor has an unidentified separate cause). Use `--no_zero_kv_sink` whenever
`--use_flash_attention` is on. Full investigation: docs/bytelm_tpu_setup.md's "zero_kv_sink +
flash-attention: investigation" section.

## Architecture

**`qcute_zero` (`qcute/qcute_zero/`, single file) — new monolithic single-shared-LM lineage,
2026-08-22**: one LM (embed + blocks) does the byte pass AND, reusing the same weights, every fuse
stage's own code-sequence NTP pass; periodic per-`Ks`-stage cross-attention ("fuse") back into the
byte stream, own weights per stage, no curriculum by design (see docs/archive5/status.md's
qcute_zero section for the full rationale/causality proof). First real `ks21`/`ks221` overfit10k runs (no
curriculum) both converged cleanly and generated coherent text — the first 3-level config in either
lineage to do so with zero curriculum. **Checkpoint caveat**: `Checkpointer` picks `best.pt` by
lowest summed total val_loss, which is a bad proxy here (that sum keeps climbing well past the point
`val_byte_acc` is still improving) — use `last.pt`, not `best.pt`, when loading a checkpoint from
this lineage for generation or further analysis, until the checkpoint metric itself is fixed.
Full-scale `ks221_1M` run (real enwik8_1M) launched 2026-08-22, in progress.

**Why `qcute_zero` avoids `qcute_v1`'s free-rollout collapse — the real differentiator**: `qcute_v1`'s
decode is structurally autoencoding at every block boundary (a block's seed token cross-attends to
*that same block's own code*, which the encoder derived from that block's own real bytes — genuinely
can't have that code before the block's bytes exist); `qcute_zero`'s decode is genuinely predictive by
construction (verified position-by-position: a byte's cross-attn mask never admits its own or any
future block's code, no exception). This is more fundamental than any weight-sharing/curriculum
detail — full analysis, plus what discipline `v1` would actually need to close the gap (a structural
fix, or a code-level consistency loss — `scheduled_sampling_p` already implements something close to
this and empirically *hurt*, not helped, an open unresolved question) in docs/archive5/status.md's
"real differentiator" section (2026-08-22). Related, current: [docs/maths.md](docs/maths.md)'s
Parts 8-9 formalize this same gap and its cost (docs/status.md's 2026-08-23 `own_code_min_lag` PoC
closes it directly, at the cost of own-block reconstruction fidelity).

**query_vec/`parallel_decode` pruned from `qcute_zero`, replaced with regular MTP heads
(2026-08-22)** — diagnosed flaw: one `query_vec` slot cost a full attention-stack pass and only
covered `parallel_decode_n_blocks` sampled clusters per step, nowhere near real MTP's density
(reuse one hidden state, many cheap linear heads). `Config.mtp_heads`/`mtp_weight` (default
`mtp_heads=1`, disabled) now add untied `nn.Linear(D, V)` heads reading the SAME final hidden state
head0's own cond/uncond readout already uses — pervasive (every position, every step), cheap (zero
extra attention FLOPs), mirroring `qcute.bytelm`'s own `mtp_heads` pattern exactly.
`generate_speculative` now drafts via these heads (one forward pass, no per-slot attention cost)
instead of `query_vec`, still verified byte-by-byte against the real, exact `generate_kv_cache`
stepper (`_make_incremental_stepper`) — same accept/reject-to-first-divergence scheme, confirmed
still **exactly** matching `generate_kv_cache`'s own trajectory after the swap. `generate_blockwise`
(the free-tier query_vec decode) was removed outright, no MTP-head replacement needed (superseded
by `generate_speculative`). The query_vec/cluster mechanism itself is preserved as its own
standalone testbed, forked onto the simpler `qcute.bytelm` trunk (no fuse-stage complexity, direct
cross-attend into the real trunk's per-layer K/V): `qcute/bytelm_queryvec/bytelm_queryvec.py`.
`qcute/qcute_zero_parallel/` (the original query-vec fork of `qcute_zero`) is left as-is, now
doubly superseded, kept only for historical reference.

**`qcute_zero`'s real incremental KV cache** (2026-08-22):
`generate_kv_cache` is no longer aliased to `generate_no_cache`'s full recompute — it's a genuine
incremental cache (byte-level self-attention + each fuse stage's post-cross-attn refinement pass,
the two `O(L)`-per-step costs; the short code-sequence/kvlm pass and fuse cross-attention itself
stay full-recompute, cheap enough not to bother caching), verified **bit-exact** identical to
`generate_no_cache` across 315 random configs via the new `check_kv_cache_consistency` diagnostic
(`qcute_zero`'s first checked-in generation-consistency check, the analog of `qcute_v1`'s
`check_roundtrip_consistency`/`check_gen_consistency`). Two real bugs were caught and fixed getting
here — full writeup in docs/archive5/status.md's "real incremental KV cache" section (2026-08-22).

**`qcute_v1` (`qcute/qcute_v1/`) is the active lineage as of 2026-08-20** — forked from a verbatim
copy of `qcute_v5` (now archived at `qcute/v5_old/`, see Commands above), implementing the
latent-AR / parallel-block-local-decode rewrite: only the top level stays a genuine NTP/AR
decoder; every level below decodes via per-block-seed-token-interleaved self-attention (a
trainable seed token — neither a true single "BOS" (it recurs every block) nor a passive "sink"
(it's a full token, not a fallback key that itself predicts nothing) — prepended before every
`K`-block, not a recurrent self-code chain) plus cross-attention to that same block's own-level
code, with the seed token's own hidden state genuinely reconstructing that block's own first byte
from that block's own code (`own_block_cross_attn_decode`/`own_block_decode_loss` in
`qcute_v1_decoder.py`, fixed
2026-08-20 — an earlier version silently never let a block's own code inform its own
reconstruction, see `docs/archive5/status.md`'s session log). Full design narrative, worked examples, and
the staged plan: [docs/qcute_v1_plan.md](docs/qcute_v1_plan.md). Progress/results log:
[docs/status.md](docs/status.md) (pruned 2026-08-23 once `qcute_zero` became a second fully-fledged
active lineage and the file passed 1300 lines — full prior history, including the v1-transition
reset, at [docs/archive5/status.md](docs/archive5/status.md),
[docs/archive4/status.md](docs/archive4/status.md), [docs/archive3/status.md](docs/archive3/status.md),
[docs/archive2/status.md](docs/archive2/status.md)). Formal bpb-validity/paradigm-comparison
writeup (not a results log): [docs/maths.md](docs/maths.md).

**TODO for a fresh session**: `configs/v1_stack_simplex/ks21_v256_pq1.py` and `ks21_v64_pq4.py`
need re-running (full-scale, `--decoder_type stack_v1` — these two configs are pinned to the now-
legacy `StackDecoderV1`, see below) — their existing `best_val_bpb` numbers in
`docs/archive5/status.md` predate the own-block-reconstruction fix above and are stale. `ks1_*` configs (`Ks=(1,)`, no
non-top level) are unaffected and don't need re-running.

**`--decoder_type` naming (2026-08-20)**: `stack` now means the current-default `StackDecoder`
(formerly `stack_v2`); the original interleaved-seed-token mechanism is `stack_v1`
(`StackDecoderV1`, now considered legacy — memory-heavier than `stack`'s non-interleaved design,
see `qcute_v1_decoder.py`'s `encode_like_self_attn_decode`/`seed_query_decode` docstrings). Also
available: `stack_local` (`StackDecoderLocal`, block-diagonal same-level conditioning) and
`stack_sync` (`StackDecoderSync`, design-note stub, `NotImplementedError` on use). Real,
incrementally-correct (not yet KV-cached) generation now works for both `stack_v1` and `stack`
(`n_levels==2` only so far) — see each class's `_generate_blockwise`/`check_blockwise_gen_consistency`.

The rest of this section (below) describes `qcute_v5`'s own architecture — still accurate for
that (frozen, archived) lineage, kept for reference when working in `qcute/v5_old/`.

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
to always `True` — see above): [docs/archive4/status.md](docs/archive4/status.md).

**Standing methodology**: use a small (`n_bytes=10000`) slice of the
corpus with a short step budget as the standard fast-iteration testbed
for architecture changes (v5 or v1) — see `configs/*_overfit10k_*.py` — until a config can actually fast-overfit
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

**Latent-AR / parallel-block-local-decode investigation**: originated here as a `qcute_v5`
window-ablation/track-dropout curriculum plan (2026-08-19); superseded by the `qcute_v1` rewrite
(2026-08-20, see top of this section) once the investigation concluded that plan's `code_window`/
lag-`D` knobs were redundant with `attn_window` and that the real fix was retargeting decode's
loss (NTP-next-block -> NTP-autoencode) plus a per-block BOS, not a track-sparsity curriculum.
Full narrative: [docs/qcute_v1_plan.md](docs/qcute_v1_plan.md).

**Non-recurrent upper-level plan (2026-08-21, not yet started)**: originates from the
`ks221`/`ks441` hard-convergence-queue collapse (repetitive single-code generation despite 96-99%+
train byte_acc, see `docs/archive5/status.md`) and a chat questioning whether the upper levels' own
autoregressive self-NTP loss is itself the cause (a constant/repeating code trivially minimizes
self-predictability, matching the observed failure signature) rather than any window/PQ/curriculum
lever tried so far. Staged plan:
1. Train a single-level (byte-only) encoder/decoder, but with many cross-attention tracks/windows
   to compensate for whatever long-range context a single level's own bounded self-attention window
   can't reach -- i.e. push long-range aggregation into decode's cross-attn window width rather than
   into an upstream recurrent summarizer (a per-chunk linear-head code, STE-trained purely from
   decode usefulness, is still fully causal even with no self-attention of its own: the hidden state
   it reads from is already causal, and `upper_track_step`'s `code_pos <= pos` rule enforces
   cross-attn causality independent of how the code was produced -- confirmed by inspection, chat
   2026-08-21).
2. Once that single-level model is converged/stable enough, freeze it, then train an upper-level LM
   on top to learn a genuine code LM (autoregressive prior over the frozen level's code stream).
3. Open question motivating step 2's own justification: is a trained upper LM actually earning its
   keep over simply re-running the (frozen) lower encoder on its own past output to get fresh codes
   on the fly -- i.e. does the upper LM do genuine long-range modeling beyond what re-encoding
   already gives for free, or is it only faster/cheaper, not more capable?

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

When explaining a result from comparing two runs/configs that differ in more than one variable,
flag causal claims as "suspect"/"maybe" rather than stating them as established — an unconfounded
isolation (one variable changed at a time) is required before a cause can be stated as fact.
