"""Overlay multiple runs' train/val bpb curves on one plot (train dashed/faint, val solid with
markers, one color per run -- consistent style so many overlaid runs stay readable/diffable),
print a sorted summary table (params, best_val_bpb, val_bpb at 25/50/100% of each run's own step
count -- a normalized convergence-rate view since runs can have different --steps budgets, final
byte_acc), and write a JSON manifest (full per-run series + summary, meant for later machine
parsing/comparison, not just eyeballing the PNG/table). Works across every module's run.jsonl
(qcute_v5* -- bpb/val_bpb; qcute.bytelm -- bpb/val_bpb; qcutelm -- bpb_total/val_bpb_total), same
auto-detection as plot_run.py.

    uv run python scripts/compare_runs.py logs/v5_word_xs logs/v5_stack_fsq_ks1_16x8
    uv run python scripts/compare_runs.py --glob "logs/v5_stack_fsq_*" --out logs/compare_fsq.png
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_KEYS = ("bpb", "bpb_total")
VAL_KEYS = ("val_bpb", "val_bpb_total")
ACC_KEYS = ("val_byte_acc", "val_head0_acc")
PARAMS_RE = re.compile(r"params=([\d.]+)M")


def load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def last_segment(records: list[dict]) -> list[dict]:
    """Same restart-dedup as plot_run.py's own last_segment -- keep only the most recent
    (final) attempt if a run_name was relaunched and its append-only logfile has old records."""
    start = 0
    prev_elapsed = -1
    for i, r in enumerate(records):
        e = r.get("elapsed_s", prev_elapsed + 1)
        if e < prev_elapsed:
            start = i
        prev_elapsed = e
    return records[start:]


def extract_series(records: list[dict], keys: tuple[str, ...]) -> tuple[list[int], list[float]]:
    steps, values = [], []
    for r in records:
        if "step" not in r:
            continue
        for k in keys:
            if k in r:
                steps.append(r["step"])
                values.append(r[k])
                break
    return steps, values


def read_params(run_dir: Path) -> float | None:
    log_path = run_dir / "run.log"
    if not log_path.exists():
        return None
    with open(log_path) as f:
        first_line = f.readline()
    m = PARAMS_RE.search(first_line)
    return float(m.group(1)) if m else None


def value_at_fraction(steps: list[int], values: list[float], frac: float) -> float | None:
    """Value of the closest logged point at or before frac * max(steps) -- a normalized
    x-axis position so runs with different --steps budgets are still comparable column-by-column."""
    if not steps:
        return None
    target = frac * max(steps)
    best_i = min(range(len(steps)), key=lambda i: abs(steps[i] - target) if steps[i] <= target + 1e-9 else float("inf"))
    return values[best_i] if steps[best_i] <= target + 1e-9 else values[0]


def resolve_run_dirs(run_paths: list[str], pattern: str | None) -> list[Path]:
    dirs = [Path(p) for p in run_paths]
    if pattern:
        dirs += [Path(p) for p in sorted(glob.glob(pattern)) if Path(p).is_dir()]
    # de-dup, keep order
    seen: set = set()
    out = []
    for d in dirs:
        if d not in seen and (d / "run.jsonl").exists():
            seen.add(d)
            out += [d]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="*", help="run directories under logs/ (e.g. logs/v5_word_xs)")
    p.add_argument("--glob", type=str, default=None, help='shell glob over run dirs, e.g. "logs/v5_stack_fsq_*"')
    p.add_argument("--out", type=Path, default=Path("logs/compare.png"))
    p.add_argument("--json", type=Path, default=None,
                    help="write a JSON manifest (summary + full series) here (default: <out>.json)")
    args = p.parse_args()
    json_path = args.json or args.out.with_suffix(".json")

    run_dirs = resolve_run_dirs(args.run_dirs, args.glob)
    if not run_dirs:
        raise SystemExit("no run dirs with a run.jsonl found (check paths/--glob)")

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    rows = []
    manifest = []

    for i, run_dir in enumerate(run_dirs):
        color = cmap(i % 10)
        records = last_segment(load_records(run_dir / "run.jsonl"))
        train_steps, train_bpb = extract_series(records, TRAIN_KEYS)
        val_steps, val_bpb = extract_series(records, VAL_KEYS)
        _, val_acc = extract_series(records, ACC_KEYS)
        if not val_steps:
            print(f"WARNING: no val bpb records in {run_dir}, skipping")
            continue

        # train dashed + faint, val solid with markers -- same color per run so the pair reads
        # as one run even when several runs are overlaid; dashed vs. solid stays distinguishable
        # in grayscale/print too, unlike relying on alpha/color alone.
        if train_steps:
            ax.plot(train_steps, train_bpb, linewidth=0.9, alpha=0.4, linestyle="--", color=color)
        ax.plot(val_steps, val_bpb, marker="o", markersize=4, linewidth=1.6, linestyle="-",
                 label=run_dir.name, color=color)

        best_val = min(val_bpb)
        summary = {
            "run": run_dir.name,
            "params_M": read_params(run_dir),
            "best_val_bpb": best_val,
            "val_bpb@25%": value_at_fraction(val_steps, val_bpb, 0.25),
            "val_bpb@50%": value_at_fraction(val_steps, val_bpb, 0.50),
            "val_bpb@100%": val_bpb[-1],
            "final_byte_acc": val_acc[-1] if val_acc else None,
            "steps": val_steps[-1],
        }
        rows += [summary]
        manifest += [{
            **summary,
            "train_steps": train_steps, "train_bpb": train_bpb,
            "val_steps": val_steps, "val_bpb": val_bpb, "val_byte_acc": val_acc,
        }]

    ax.set_xlabel("step")
    ax.set_ylabel("bits per byte")
    ax.set_title("bpb comparison (dashed=train, solid=val)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")
    ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    rows.sort(key=lambda r: r["best_val_bpb"])
    manifest.sort(key=lambda r: r["best_val_bpb"])
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {json_path}\n")

    cols = ["run", "params_M", "best_val_bpb", "val_bpb@25%", "val_bpb@50%", "val_bpb@100%", "final_byte_acc", "steps"]
    widths = {c: max(len(c), *(len(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols}

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(fmt(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
