#!/usr/bin/env python3
"""Vane - static figures for the README.

Read-only. Every figure is built from CSVs the CLI already produces, and
nothing here touches the pricing, simulation, model or evaluation logic. If a
required file is missing the figure is skipped with a note saying which command
produces it, so this script never silently invents data.

    make figures
    python3 analysis/make_figures.py --out docs

Inputs, and the commands that produce them:

    data/train_events.csv   ./build/vane-sim --weeks 26 --jitter 0.22 --out data/train
    data/train_oracle.csv   (same command)
    data/backtest.csv       ./build/vane-backtest --model models/logistic.txt \
                                --seeds 5 --ablate --csv data/backtest.csv
    data/phase4_loop.csv    python3 analysis/run_phase4.py --generations 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI or a headless box
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Muted, print-friendly, and distinguishable in greyscale.
INK = "#1b1b1b"
GREY = "#8c8c8c"
BLUE = "#2c5f8a"
RUST = "#b4472e"
GREEN = "#4a7c59"
SAND = "#c9a227"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "legend.frameon": False,
    "grid.color": "#e2e2e2",
    "grid.linewidth": 0.7,
})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, out: Path, name: str) -> str:
    # Both formats: PNG is what the README embeds, because GitHub renders it
    # identically everywhere including the mobile app, while SVG is kept for
    # slides and print where it stays sharp at any size.
    fig.savefig(out / f"{name}.svg", format="svg")
    fig.savefig(out / f"{name}.png", format="png", dpi=150)
    plt.close(fig)
    return f"{name}.png"


def _need(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


# ---------------------------------------------------------------------------
def fig_extrapolation(out: Path) -> str | None:
    """The Phase 3 failure: a tree model going flat outside its data."""
    from vane_data import load, time_split
    from vane_demand import BoostedDemand, LogisticDemand
    from vane_support import support_bounds

    src = ROOT / "data/train_events.csv"
    if not _need(src, ROOT / "data/train_oracle.csv"):
        return None

    ds = load(str(ROOT / "data/train"))
    tr, te = time_split(ds, 0.7)
    y = tr.events["accepted"].to_numpy()
    bounds = support_bounds(tr.events)
    lo, hi = bounds["retail"]

    lg = LogisticDemand().fit(tr.events, y)
    bs = BoostedDemand().fit(tr.events, y)

    row = te.events[te.events["tier"] == "retail"].iloc[[0]]
    oid = row["id"].iloc[0]
    o = te.oracle[te.oracle["id"] == oid].iloc[0]

    grid = np.linspace(100, 620, 220)
    rep = row.loc[row.index.repeat(len(grid))]
    p_lg = lg.predict_proba(rep, spread=grid)
    p_bs = bs.predict_proba(rep, spread=grid)
    p_true = norm.cdf((o["mu"] - np.log(grid)) / o["sigma_ln"])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1.35, 1]})

    ax.axvspan(grid[0], hi, color="#f2f4f6", zorder=0)
    ax.axvline(hi, color=GREY, lw=0.9, ls=(0, (4, 3)))
    ax.text(hi - 6, 0.95, "logged support ends", ha="right", va="top",
            fontsize=8, color=GREY)

    ax.plot(grid, p_true, color=INK, lw=2.0, label="true demand")
    ax.plot(grid, p_lg, color=BLUE, lw=1.6, label="logistic")
    ax.plot(grid, p_bs, color=RUST, lw=1.6, label="boosted (monotone)")
    ax.set_ylabel("P(accept)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.set_title("Beyond its data, the tree model stops falling")
    _grid(ax)

    cost = 1.5
    ax2.axvspan(grid[0], hi, color="#f2f4f6", zorder=0)
    ax2.axvline(hi, color=GREY, lw=0.9, ls=(0, (4, 3)))
    for series, colour, label in ((p_true, INK, "true"), (p_lg, BLUE, "logistic"),
                                  (p_bs, RUST, "boosted")):
        rev = (grid - cost) * series
        ax2.plot(grid, rev, color=colour, lw=1.6, label=label)
        ax2.plot(grid[np.argmax(rev)], rev.max(), "o", color=colour, ms=5)
    ax2.set_xlabel("quoted half-spread (bps)")
    ax2.set_ylabel("expected margin (bps)")
    ax2.set_title("so revenue keeps climbing, and the policy chases it out")
    _grid(ax2)

    fig.tight_layout()
    return _save(fig, out, "extrapolation")


def fig_support_collapse(out: Path) -> str | None:
    """The Phase 4 result: a deployed policy narrowing its own training data."""
    src = ROOT / "data/phase4_loop.csv"
    if not _need(src):
        return None
    df = pd.read_csv(src)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))

    for tag, colour, label in (("decay", RUST, "exploration decays"),
                               ("floor", GREEN, "exploration held at a floor")):
        d = df[df["tag"] == tag].sort_values("generation")
        if d.empty:
            continue
        ax.plot(d["generation"], d["retail_support_width"], "o-", color=colour,
                lw=1.8, ms=5, label=label)
        ax2.plot(d["generation"], d["regret_policy"], "o-", color=colour, lw=1.8, ms=5)

    ax.set_xlabel("retraining generation")
    ax.set_ylabel("retail support width (bps)")
    ax.set_title("The data the model can speak about")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", fontsize=8)
    _grid(ax)

    ax2.set_xlabel("retraining generation")
    ax2.set_ylabel("true regret (bps per unit)")
    ax2.set_title("What that costs")
    ax2.set_ylim(bottom=0)
    _grid(ax2)

    fig.suptitle("Each generation trains on the data the previous one produced",
                 fontsize=10, fontweight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, out, "support_collapse")


def fig_regret_by_tier(out: Path) -> str | None:
    """The Phase 2/3 finding: the static table is wrong in both directions."""
    from vane_data import load, time_split
    from vane_demand import LogisticDemand
    from vane_policy import by_tier
    from vane_support import optimal_spread_in_support, support_bounds

    if not _need(ROOT / "data/train_events.csv", ROOT / "data/train_oracle.csv"):
        return None

    ds = load(str(ROOT / "data/train"))
    tr, te = time_split(ds, 0.7)
    model = LogisticDemand().fit(tr.events, tr.events["accepted"].to_numpy())
    bounds = support_bounds(tr.events)
    pol = optimal_spread_in_support(model, te.events, bounds)
    t = by_tier(te.events, te.oracle, pol)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.2))
    x = np.arange(len(t))
    w = 0.26

    ax.bar(x - w, t["logged_bps"], w, color=GREY, label="static table", zorder=3)
    ax.bar(x, t["learned_bps"], w, color=BLUE, label="learned", zorder=3)
    ax.bar(x + w, t["oracle_bps"], w, color=INK, label="oracle", zorder=3)
    ax.set_xticks(x, t["tier"])
    ax.set_ylabel("mean half-spread (bps)")
    ax.set_title("Mispriced in both directions")
    ax.legend(fontsize=8)
    _grid(ax)

    # Retail is over-charged, corporate under-charged: annotate the direction.
    for i, r in t.reset_index().iterrows():
        gap = r["logged_bps"] - r["oracle_bps"]
        ax.annotate(f"{gap:+.0f}", (i - w, r["logged_bps"]), ha="center",
                    va="bottom", fontsize=7.5,
                    color=RUST if gap > 0 else GREEN)

    ax2.bar(x - w / 2, t["regret_logged"], w, color=GREY, label="static table", zorder=3)
    ax2.bar(x + w / 2, t["regret_learned"], w, color=BLUE, label="learned", zorder=3)
    ax2.set_xticks(x, t["tier"])
    ax2.set_ylabel("regret vs oracle (bps per unit)")
    ax2.set_title("Margin left on the table")
    ax2.legend(fontsize=8)
    _grid(ax2)

    fig.tight_layout()
    return _save(fig, out, "regret_by_tier")


def fig_backtest(out: Path) -> str | None:
    """Phase 4: the learned policy wins on every world, and the ablations."""
    src = ROOT / "data/backtest.csv"
    if not _need(src):
        return None
    df = pd.read_csv(src)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.2))

    piv = df[df["policy"].isin(["static", "learned"])].pivot(
        index="seed", columns="policy", values="total_pnl") / 1e6
    x = np.arange(len(piv))
    # Bars start below the data rather than at zero: every policy makes money,
    # so a zero baseline wastes the whole panel showing that they are all tall.
    floor = float(piv.min().min()) * 0.96
    ax.bar(x - 0.18, piv["static"] - floor, 0.36, bottom=floor, color=GREY,
           label="static", zorder=3)
    ax.bar(x + 0.18, piv["learned"] - floor, 0.36, bottom=floor, color=BLUE,
           label="learned", zorder=3)
    ax.set_ylim(floor, float(piv.max().max()) * 1.06)
    ax.set_xticks(x, [f"seed {s}" for s in piv.index], fontsize=8)
    ax.set_ylabel("total PnL (millions, quote ccy)")
    ax.set_title("Same price path, same customers")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.0), ncols=2)
    _grid(ax)
    lift = 100 * (piv["learned"].mean() / piv["static"].mean() - 1)
    ax.annotate(f"{lift:+.1f}% mean", xy=(0.5, -0.24), xycoords="axes fraction",
                ha="center", fontsize=9, color=BLUE, fontweight="bold")

    # Ablations: PnL against dispersion across seeds, since the risk controls
    # show up in the spread of outcomes more than in the mean.
    order = ["static", "learned", "learned-no-skew", "learned-no-volterm",
             "learned-no-hedge"]
    present = [p for p in order if p in set(df["policy"])]
    g = df[df["policy"].isin(present)].groupby("policy")["total_pnl"]
    means = (g.mean() / 1e6).reindex(present)
    sds = (g.std() / 1e6).reindex(present).fillna(0.0)
    colours = {"static": GREY, "learned": BLUE, "learned-no-skew": SAND,
               "learned-no-volterm": GREEN, "learned-no-hedge": RUST}
    labels = {"static": "static", "learned": "learned",
              "learned-no-skew": "no inventory skew",
              "learned-no-volterm": "no volatility term",
              "learned-no-hedge": "no hedging"}
    y = np.arange(len(present))
    left = max(0.0, float((means - sds).min()) * 0.9)
    ax2.barh(y, means - left, 0.6, left=left, xerr=sds,
             color=[colours[p] for p in present],
             error_kw={"ecolor": INK, "elinewidth": 1.0, "capsize": 3}, zorder=3)
    ax2.set_xlim(left, float((means + sds).max()) * 1.05)
    ax2.set_yticks(y, [labels[p] for p in present], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("mean PnL (millions); bars are sd across seeds")
    ax2.set_title("Removing one risk control at a time")
    _grid(ax2, axis="x")

    fig.tight_layout()
    return _save(fig, out, "backtest")


def fig_weekend(out: Path) -> str | None:
    """Phase 1: the weekend widener, derived rather than bolted on."""
    if not _need(ROOT / "data/train_events.csv"):
        return None
    ev = pd.read_csv(ROOT / "data/train_events.csv")
    ev = ev[ev["tier"] == "retail"]
    if ev.empty:
        return None

    ev = ev.assign(slot=(ev["weekday"] * 24 + ev["hour_utc"]).round(0))
    g = ev.groupby("slot").agg(spread=("quoted_half_bps", "mean"),
                               horizon=("horizon_h", "mean"),
                               n=("id", "size"))
    g = g[g["n"] >= 5]

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.axvspan(4 * 24 + 22, 6 * 24 + 22, color="#f2f4f6", zorder=0)
    ax.text(5 * 24 + 22, g["spread"].max(), "interbank market shut", ha="center",
            va="top", fontsize=8, color=GREY)
    ax.plot(g.index, g["spread"], color=BLUE, lw=1.5)
    ax.set_xticks([d * 24 + 12 for d in range(7)],
                  ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlim(0, 168)
    ax.set_ylabel("mean retail half-spread (bps)")
    ax.set_title("The weekend widener falls out of the hedging horizon, "
                 "not a special case")
    _grid(ax)
    fig.tight_layout()
    return _save(fig, out, "weekend_widener")


FIGURES = [
    ("extrapolation", fig_extrapolation,
     "./build/vane-sim --weeks 26 --jitter 0.22 --out data/train"),
    ("support_collapse", fig_support_collapse,
     "python3 analysis/run_phase4.py --generations 5"),
    ("regret_by_tier", fig_regret_by_tier,
     "./build/vane-sim --weeks 26 --jitter 0.22 --out data/train"),
    ("backtest", fig_backtest,
     "./build/vane-backtest --model models/logistic.txt --seeds 5 --ablate "
     "--csv data/backtest.csv"),
    ("weekend_widener", fig_weekend,
     "./build/vane-sim --weeks 26 --jitter 0.22 --out data/train"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    ap.add_argument("--only", default="", help="build a single figure by name")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    made, skipped = [], []
    for name, fn, hint in FIGURES:
        if args.only and args.only != name:
            continue
        result = fn(out)
        if result:
            made.append(result)
            print(f"  wrote {args.out}/{result}")
        else:
            skipped.append((name, hint))
            print(f"  skip  {name} (missing input)")

    if skipped:
        print("\nto produce the missing inputs:")
        for name, hint in skipped:
            print(f"  {name:18s} {hint}")
    print(f"\n{len(made)} figures written to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
