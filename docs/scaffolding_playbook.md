# Scaffolding playbook

A reusable, detailed prompt for bootstrapping a new uv-managed, design-doc-
driven ML research project the way *this* repo (qcute) is actually built —
not just the initial skeleton, but the experiment-running conventions it
converged on. Hand this to a fresh Claude Code session (swap in the new
design doc, package name, and dataset) to reproduce the structure and style.

---

Set up this repo as a uv-managed Python project implementing the design in
`<DESIGN_DOC>.md`. Match this exact structure and these conventions:

## 1. Environment

`uv init --package --python 3.12`. If a package dir already exists,
consolidate into it (flat layout, not `src/`) — set
`[tool.uv.build-backend] module-root = ""` for flat layout, or
`[tool.uv] package = false` if this is just runnable scripts, not something
meant to be installed/importable elsewhere (qcute uses `package = false`).

## 2. `.gitignore` up front

venv, caches, build artifacts, and every directory that will fill up with
run artifacts: `datasets/` (downloaded/derived data), `logs/` (training
logs — see §6). Add these *before* the first training run, not after.

## 3. One module per component, named after its main class

Implement each major component from the design doc as its own module
inside the package — e.g. `qcute/bytelm.py` defines `class ByteLM`,
`qcute/qcutelm.py` defines `class QCuteLM`. **The module filename should
match the primary class it defines** (lowercased) — this makes `from
pkg.bytelm import ByteLM` self-explanatory and avoids the alternative of
generic names like `model.py` that don't survive a second model being
added later.

Never leave a bare top-level script with the same name as the package dir
(e.g. `foo.py` next to `foo/` silently breaks `import foo` for anything
else in the repo). Each module should be genuinely self-contained — no
imports between sibling modules unless there's a concrete reuse need (qcute's
`bytelm.py` and `qcutelm.py` duplicate a small `Logger` class and
`load_enwik8`/`lr_at` helpers rather than share them, on purpose: it keeps
each file readable standalone and avoids premature coupling between things
that are still actively changing). Run everything via `python -m pkg.module`.

Superseded implementations get moved to `archive/<descriptive_name>.py`
(not deleted) with a one-line header explaining what replaced them and why
— `git mv`, not `rm`, so history is preserved.

## 4. Every training script's `main()` follows the same shape

This is the part worth copying exactly, because it's what makes running
many small experiments tractable:

**a. Two-tier config: Python config file + CLI override.**
Add a `--config path/to/file.py` flag, parsed in a pre-pass (`parse_known_args`
on a small parser with just `--config`) before building the full argument
parser. Load the config file as a plain Python module (`importlib.util`,
not YAML/JSON — real Python means `Path(...)`, arithmetic, and comments are
free) and feed its module-level variables into `parser.set_defaults(...)`,
filtered to known argument names. Then do the real `parser.parse_args()`.
Net effect: **config file values override the script's hardcoded defaults,
but explicit CLI flags override the config file.** This gives you
reproducible named experiments (`configs/bytelm_xs_mtp4_converged.py`) without
losing the ability to override one flag ad hoc for a quick variant.
Config files live in `configs/`, one per named experiment, with a docstring
explaining what the experiment tests and the exact invocation.

**b. Run naming falls back through three tiers**, most to least specific:
explicit `--run_name` → else the `--config` file's stem → else a formula
from the CLI args themselves (e.g. `bytelm_<preset>_<timestamp>`).
Implement this as a plain if/elif/else in `main()`, not magic. Give every
run its own directory keyed by this name — `logs/<run_name>/` and
`checkpoints/<run_name>/` — rather than flat files distinguished only by a
long filename; it's what makes "find everything about run X" a single
`ls`, and what lets the log and checkpoint file names themselves be boring
(`run.log`, `best.pt`) since the directory already carries the identity.

**c. Every run gets a `tqdm` progress bar *and* two logfiles**, both written
only at `--log_every`/`--eval_every` intervals — never on every step, and
never touched by tqdm's live `\r`-redraws:
  - `<name>.log` — raw human-readable text, exactly what's printed to the
    terminal (via `tqdm.write()`, which doesn't clobber the progress bar).
    `tail -f` this from another terminal to watch a background run live.
  - `<name>.jsonl` — one JSON object per line, same information
    structured, for later plotting/analysis.

Every logged line is prefixed with elapsed time as `[HH:MM:SS]` (tracked
from when the `Logger` was constructed), and the JSON record carries
`elapsed_s` (int) and `elapsed_hms` alongside the metrics — not a raw Unix
epoch timestamp, which isn't human-readable and isn't what anyone actually
wants when reading a log back later.

Implement this as a small `Logger` class, duplicated per-script like
everything else in §3. Two details worth calling out because they're easy
to get wrong on a rewrite:

- The `.log` file *is* "raw stdout, filtered to just the interval lines" —
  don't reach for an OS-level `| tee | grep` pipeline to get that; tqdm's
  bar already goes to stderr by default, and writing `line` here at the
  same cadence as the terminal print is the filtering, done once, not
  re-derived from a noisy captured stream after the fact.
- In the JSON record, drop `msg` whenever `record` (the structured kwargs)
  is non-empty — the string is just the parsed-out fields re-serialized as
  text, so keeping both duplicates every number in the file twice. Keep
  `msg` only for plain informational lines that carry no structured data
  (e.g. `log(f"train_bytes={n}")` with no kwargs) — those would otherwise
  vanish from the JSON entirely.

```python
class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.text_path = path.with_suffix(".log")
        self.json_path = path.with_suffix(".jsonl")
        self.text_f = open(self.text_path, "a")
        self.json_f = open(self.json_path, "a")
        self.start_time = time.time()

    def __call__(self, msg: str, **record) -> None:
        elapsed_s = int(time.time() - self.start_time)
        line = f"[{format_hms(elapsed_s)}] {msg}"
        tqdm.write(line)
        self.text_f.write(line + "\n"); self.text_f.flush()
        json_record = {"elapsed_s": elapsed_s, **({} if record else {"msg": msg}), **record}
        self.json_f.write(json.dumps(json_record) + "\n"); self.json_f.flush()
```

**d. Train/val split + periodic eval, always**, even for a quick script —
`split_train_val(data, val_frac)` and an `eval_*()` function run every
`--eval_every` steps on a held-out slice. Report both train-step metrics
(noisy, every `--log_every`) and val metrics (averaged over `--eval_batches`,
less noisy) — the gap between them is often the most important signal (see
§7).

## 5. Data prep is its own idempotent script

`scripts/prepare_data.py`: downloads the full dataset if missing (skips if
present, `--force` to re-fetch), and cuts a small `_tiny` variant (e.g.
500,000 bytes) for fast local iteration — every training script accepts
`--data path/to/either` so the same code path exercises both scales.
Running the script twice does nothing destructive the second time.

## 6. Long/background runs: `run_in_background` + a `Monitor` filter, never a sleep-poll loop

For anything that'll run more than ~2 minutes: launch it, let it move to
background automatically, then attach a `Monitor` that tails the *text*
logfile and greps for the lines worth surfacing (e.g. `val_bpb` checkpoints)
**plus failure signatures** (`Traceback|Error|Killed`) so a crash isn't
silently missed. Never `sleep`-loop and poll manually — either wait for the
background task's own completion notification, or let the Monitor's
grep-filtered tail deliver periodic updates. If you need to change
hyperparameters mid-run, there's no checkpoint/resume in scripts this
simple — stop the run and relaunch with a bigger step budget; because the
LR schedule is stateless (see §7), this is equivalent to having let the
original run continue, not a wasted restart.

## 7. LR schedule: linear warmup, then constant — deliberately, for comparability

Use `lr_at(step, warmup, peak) = peak * step/warmup` during warmup, `peak`
after — not cosine decay. This is a real methodological choice, not just
"simpler": a non-decaying LR is the standard way to *expose* whether a
model is overfitting a small dataset (train loss keeps dropping while val
rises, sometimes non-monotonically — this is the empirical signature of
epoch-wise double descent, Nakkiran et al. 2019, and it's specifically
easiest to see under a constant LR because decay would suppress it). Use
the exact same schedule function/params across every script being compared
head-to-head — otherwise a BPB difference could just be an LR-schedule
artifact, not a real architectural finding.

## 8. `docs/` structure

- The design doc itself, moved in verbatim.
- `architecture.md`: maps code → doc sections, one `##` per module,
  documents load-bearing non-obvious fixes (`"don't remove this init
  scaling, default init blows up the loss to ~1000 bits/byte"`), and a
  "known gaps vs. the full design" section at the end.
- `status.md`: phase-by-phase checklist against the design doc's own phase
  plan, updated as work lands — not aspirational, an honest record of
  what's actually been run vs. just implemented-but-untested.
- This file.

## 8b. Plotting: one script, reads the JSONL, PNG per run

`scripts/plot_run.py logs/<run_name>` reads `run.jsonl`, auto-detects
whichever train/val metric keys are present (different scripts may name
theirs differently — don't force one schema across scripts that measure
different things), and writes `<run_dir>/bpb.png`. Two things matter for
readability, not just correctness:

- **Size the val marker to its sampling rate, not a fixed default.** Val is
  evaluated far less often than train is logged (`--eval_every >>
  --log_every`), so a same-size marker makes val basically invisible against
  a denser train line. A visibly bigger val marker (not just a different
  color) is what makes "where did val bottom out, before it started rising
  again" readable at a glance — which is usually the one plot you actually
  need after a run, i.e. "did we overfit, and when."
- **Detect and drop stale segments from restarted runs.** Because logging
  (§4c) is append-only and keyed by `run_name`, stopping a run and relaunching
  with the *same* `run_name` appends a second run's records to the same
  file. Plotted naively as one connected line, this zigzags backward in
  step at the restart boundary (visually "borked" — jumps forward then back
  then forward). Detect restarts by watching `elapsed_s` for a decrease
  from the previous record, and keep only the last segment before plotting.
  This is a direct consequence of the append-only logging choice — expect
  it, don't be surprised by it, and handle it in the plot script rather
  than treating every experiment as needing a fresh `run_name`.

## 8c. Verify "exact" empirically — don't just trust the theory

If a metric's docstring says "exact, not an estimate," write the
verification, don't just derive it on paper and assume the library behaves
the way the derivation assumes. Concretely (this repo's example): a BPE
tokenizer's bits-per-byte should be exactly `sum(token_nats) /
sum(token_byte_lengths) / ln(2)` if — and only if — encode/decode is a true
bijection with the original bytes. It wasn't: the tokenizer library's
NLP-oriented defaults (Unicode normalization, whitespace collapsing, a
synthetic leading-space prefix) silently dropped information, so the
"exact" formula was computing something exact for a *different*, lossily-
normalized string, not the original corpus. This surfaces as a fixable
config problem (disable normalization, disable whitespace collapsing,
disable the dummy prefix, add byte-fallback for full coverage) — but only
if you actually run `sum(byte_len_table[ids]) == len(original_bytes)` and
`decode(encode(text)) == text` as real assertions, not skip straight to
"the math works out, ship it." Put the check in the training script itself
(hard-fail if it doesn't hold) so a future re-run with a config that breaks
losslessness fails loudly instead of silently producing a wrong number.

## 8d. Don't hardcode a bandwidth/compute target without checking it's reachable at your data scale

If your design doc specifies a target (e.g. "aim for N units of
bandwidth/compression per step") calibrated for full-scale training, and
you're building/debugging at a much smaller data scale for fast iteration,
*check whether that target is even achievable* at the smaller scale before
wiring it in as a default. Concretely: a design's full-corpus BPE
recommendation (~8 bytes/token) turned out to require vocabulary sizes that,
on a 500KB debug-scale corpus, just memorize phrases instead of generalizing
— the target wasn't wrong, it was scale-mismatched. The fix was calibrating
the small-scale presets (`xs`, tiny-corpus defaults) to a target actually
reachable at that data size (4, not 8), documented as scale-specific
(comments pointing at *why*, referencing the corpus-size ceiling), while
keeping the full-scale presets (`sd`/`md`) at the doc's original number.
One config knob serving two different scales convincingly is rare — prefer
naming the mismatch explicitly over pretending one number fits both.

## 9. `CLAUDE.md`: lean, always

Commands block + a two-sentence architecture pointer into
`docs/architecture.md`. Never duplicate `architecture.md`'s content here —
one is the fast-glance summary, the other is the detail; if they drift the
lean one wins on staleness risk.

## 10. Workflow policy

Implement continuously without pausing for review checkpoints — the user
watches diffs/terminal output live and interrupts to redirect. Smoke-test
every script with tiny `--steps`/`--n_bytes` before calling it done — this
is how a bad default init producing nonsense loss on step 1 gets caught
immediately instead of after a 30-minute run. Never `git commit`/`git push`
unless explicitly asked for that specific commit/push.

## 11. Claude response style — put it in `CLAUDE.md`, not just this file

Session-specific conventions about *how Claude should reply* belong in
`CLAUDE.md` under a `## Response format` heading, so they persist and apply
automatically in future sessions rather than being re-stated each time —
e.g. this repo's convention: prefix every reply with the current timestamp
(`date`, never guessed). When a monitored background run is in progress,
keep updates on new checkpoints short — one or two sentences stating the
new value, the trend vs. the last checkpoint, and whether it changes
anything actionable; don't re-derive or repeat prior analysis each time.
