"""Plot train/val bpb from a run's run.jsonl to a PNG.

    uv run python scripts/plot_run.py logs/<run_name>
    uv run python scripts/plot_run.py logs/<run_name>/run.jsonl --out somewhere/else.png

Works across qcute.bytelm ("bpb"/"val_bpb"), qcute.qcutelm ("bpb_total"/
"val_bpb_total"), and qcute.bpelm ("bpb"/"val_bpb") — auto-detects whichever
key is present per JSONL record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_KEYS = ("bpb", "bpb_total")
VAL_KEYS = ("val_bpb", "val_bpb_total")


def load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_path", type=Path, help="path to run.jsonl, or the run directory containing it")
    p.add_argument("--out", type=Path, default=None, help="output PNG (default: <run_dir>/bpb.png)")
    args = p.parse_args()

    jsonl_path = args.run_path / "run.jsonl" if args.run_path.is_dir() else args.run_path
    records = load_records(jsonl_path)

    train_steps, train_bpb = extract_series(records, TRAIN_KEYS)
    val_steps, val_bpb = extract_series(records, VAL_KEYS)
    if not train_steps and not val_steps:
        raise SystemExit(f"no bpb records found in {jsonl_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_steps, train_bpb, linewidth=1, alpha=0.7, label="train bpb", color="#4C72B0")
    # val is logged far less often than train (eval_every >> log_every) — a
    # bigger marker keeps the sparse points visually trackable against the
    # denser train line, rather than disappearing into it.
    ax.plot(
        val_steps, val_bpb, marker="o", markersize=8, linewidth=1.5,
        label="val bpb", color="#DD8452",
    )

    ax.set_xlabel("step")
    ax.set_ylabel("bits per byte")
    ax.set_title(jsonl_path.parent.name)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = args.out or jsonl_path.parent / "bpb.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
