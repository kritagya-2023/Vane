#!/usr/bin/env python3
"""Tests for the figure layer.

Deliberately thin. Figures are presentation, not logic, and asserting on pixels
would be brittle without catching anything that matters. What is worth pinning
is narrower:

  - the script is read-only, and cannot alter any input
  - a missing input is skipped with a usable hint, never silently faked
  - the columns each figure depends on actually exist in the CLI's output
  - every figure produces a non-trivial SVG when its inputs are present

    python3 analysis/test_figures.py
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent

CHECKS = {"n": 0, "failed": 0}


def check(cond: bool, msg: str) -> None:
    CHECKS["n"] += 1
    if not cond:
        CHECKS["failed"] += 1
        print(f"  FAIL  {msg}")


def section(name: str) -> None:
    print(f"\n-- {name}")


def digest(paths) -> dict:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths if p.exists()
    }


def test_contracts() -> None:
    section("input contracts")
    import pandas as pd

    # Each figure reads specific columns out of the CLI's CSVs. If a column is
    # renamed upstream this fails here rather than in a broken chart.
    expect = {
        "data/train_events.csv": {"tier", "quoted_half_bps", "weekday", "hour_utc",
                                  "horizon_h", "accepted", "id"},
        "data/backtest.csv": {"policy", "seed", "total_pnl"},
        "data/phase4_loop.csv": {"tag", "generation", "retail_support_width",
                                 "regret_policy"},
    }
    for rel, cols in expect.items():
        path = ROOT / rel
        if not path.exists():
            print(f"  SKIP  {rel} not present")
            continue
        have = set(pd.read_csv(path, nrows=1).columns)
        missing = cols - have
        check(not missing, f"{rel} is missing columns: {sorted(missing)}")

    # The closed-loop file must carry both arms, or the comparison figure is
    # silently half a chart.
    loop = ROOT / "data/phase4_loop.csv"
    if loop.exists():
        tags = set(pd.read_csv(loop)["tag"])
        check({"decay", "floor"} <= tags, "phase4 loop has both decay and floor arms")


def test_missing_inputs_are_skipped() -> None:
    section("missing inputs degrade gracefully")
    import make_figures as mf

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # Point every figure at a directory with no data by temporarily moving
        # ROOT. Simpler: call a figure whose input certainly does not exist.
        real_root = mf.ROOT
        try:
            mf.ROOT = Path(tmp) / "nowhere"
            for name, fn, hint in mf.FIGURES:
                result = fn(out)
                check(result is None, f"{name} returns None when its input is absent")
                check(bool(hint), f"{name} carries a hint for producing its input")
            check(not list(out.glob("*.svg")), "no figure files written without inputs")
        finally:
            mf.ROOT = real_root


def test_figures_render() -> None:
    section("figures render when inputs are present")
    import make_figures as mf

    inputs = [ROOT / "data/train_events.csv", ROOT / "data/backtest.csv",
              ROOT / "data/phase4_loop.csv"]
    before = digest(inputs)
    if not before:
        print("  SKIP  no input data; run make figures first")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        made = 0
        for name, fn, _ in mf.FIGURES:
            result = fn(out)
            if result is None:
                print(f"  SKIP  {name} (input absent)")
                continue
            made += 1
            png = out / result
            svg = png.with_suffix(".svg")
            check(png.exists(), f"{name} wrote a PNG")
            check(svg.exists(), f"{name} wrote an SVG")
            body = svg.read_text()
            # A blank axes frame is around 8 KB; anything real is much larger.
            check(len(body) > 15_000, f"{name} produced a non-trivial figure")
            check(body.lstrip().startswith("<?xml"), f"{name} produced valid SVG")
            check("<path" in body, f"{name} drew something")
            check(png.stat().st_size > 20_000, f"{name} PNG is non-trivial")
            check(png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{name} PNG is valid")
        check(made > 0, "at least one figure rendered")

    # Read-only: the figure layer must not touch the data it visualises.
    check(digest(inputs) == before, "inputs are unchanged after rendering")


def test_no_core_imports_mutated() -> None:
    section("read-only over the core")
    src = (ROOT / "analysis/make_figures.py").read_text()

    # The figure layer may read the analysis modules, but must not write CSVs,
    # export models, or shell out to the simulator.
    for bad in ("to_csv(", "subprocess", "os.remove", "shutil", "open(", "unlink"):
        check(bad not in src, f"make_figures.py does not use {bad!r}")
    check("matplotlib.use(\"Agg\")" in src, "renders headless")


def main() -> int:
    test_contracts()
    test_missing_inputs_are_skipped()
    test_figures_render()
    test_no_core_imports_mutated()
    print(f"\nfigures: {CHECKS['n']} checks, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
