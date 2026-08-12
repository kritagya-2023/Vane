#!/usr/bin/env python3
"""Export a fitted demand model for the C++ serving path.

Phase 3 fitted the model in Python. Phase 4 runs it inside the simulator, which
is C++. Rather than embed a Python interpreter in the hot path, the fitted
coefficients are written to a small text file that the engine reads at startup
-- the same split a real desk uses between a research stack and a serving one.

The file carries the feature names alongside the numbers, and the loader
asserts they match the order the engine expects. A reordering in `FEATURES`
then fails loudly at load instead of silently mispricing every quote.

    python3 analysis/export_model.py --data data/train --out models/logistic.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vane_data import FEATURES, load, time_split  # noqa: E402
from vane_demand import LogisticDemand  # noqa: E402
from vane_support import support_bounds  # noqa: E402


def export(prefix: str, out_path: str, train_frac: float = 1.0) -> dict:
    ds = load(prefix, with_oracle=False)
    if train_frac < 1.0:
        ds, _ = time_split(ds, train_frac)

    y = ds.events["accepted"].to_numpy()
    model = LogisticDemand().fit(ds.events, y)
    bounds = support_bounds(ds.events)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# vane logistic demand model\n")
        f.write(f"# fitted on {len(ds)} events from {prefix}\n")
        f.write("# feature <name> <mean> <scale> <coef>\n")
        f.write(f"intercept {model.model.intercept_[0]:.12g}\n")
        for i, name in enumerate(FEATURES):
            f.write(
                f"feature {name} {model.scaler.mean_[i]:.12g} "
                f"{model.scaler.scale_[i]:.12g} {model.model.coef_[0][i]:.12g}\n"
            )
        for tier, (lo, hi) in sorted(bounds.items()):
            f.write(f"support {tier} {lo:.6f} {hi:.6f}\n")

    return {"n": len(ds), "bounds": bounds, "path": str(out)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="models/logistic.txt")
    ap.add_argument("--train-frac", type=float, default=1.0)
    args = ap.parse_args()

    info = export(args.data, args.out, args.train_frac)
    print(f"exported {info['path']} from {info['n']} events")
    for tier, (lo, hi) in sorted(info["bounds"].items()):
        print(f"  support {tier:10s} {lo:7.1f} - {hi:7.1f} bps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
