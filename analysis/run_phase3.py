#!/usr/bin/env python3
"""Vane Phase 3 - the experiment.

Fits demand models on logged quote outcomes, derives a pricing policy from
each, and scores them three ways: against the truth the simulator knows,
against off-policy estimators that a real desk could compute, and against an
audit of how far each policy strays outside its data.

    python3 analysis/run_phase3.py --data data/train
    python3 analysis/run_phase3.py --data data/train --seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vane_data import load, time_split  # noqa: E402
from vane_demand import BoostedDemand, LogisticDemand, evaluate  # noqa: E402
from vane_policy import (  # noqa: E402
    off_policy_value,
    optimal_spread,
    oracle_regret,
    by_tier,
)
from vane_support import (  # noqa: E402
    extrapolation_audit,
    optimal_spread_in_support,
    support_bounds,
)

FMT = lambda x: f"{x:8.2f}"  # noqa: E731


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def run_one(prefix: str, train_frac: float = 0.7, quiet: bool = False) -> dict:
    ds = load(prefix)
    if ds.oracle is None:
        raise SystemExit(f"{prefix}_oracle.csv not found; scoring needs it")
    tr, te = time_split(ds, train_frac)
    ytr = tr.events["accepted"].to_numpy()
    yte = te.events["accepted"].to_numpy()
    bounds = support_bounds(tr.events)

    if not quiet:
        rule("data")
        print(f"  train {len(tr):6d} events   {tr.events.t_sec.min()/86400:6.1f} - "
              f"{tr.events.t_sec.max()/86400:6.1f} days")
        print(f"  test  {len(te):6d} events   {te.events.t_sec.min()/86400:6.1f} - "
              f"{te.events.t_sec.max()/86400:6.1f} days")
        print(f"  accept rate: train {ytr.mean():.3f}  test {yte.mean():.3f}")
        print("\n  logged spread support used by the constrained policy:")
        for t, (lo, hi) in sorted(bounds.items()):
            print(f"    {t:10s} {lo:6.1f} - {hi:6.1f} bps")

    results = {}
    for name, ctor in [("logistic", LogisticDemand), ("boosted", BoostedDemand)]:
        model = ctor().fit(tr.events, ytr)
        fit = evaluate(model, te.events, yte)

        free = optimal_spread(model, te.events)
        cons = optimal_spread_in_support(model, te.events, bounds)
        r_free = oracle_regret(te.events, te.oracle, free)
        r_cons = oracle_regret(te.events, te.oracle, cons)
        ope = off_policy_value(te.events, cons, model=model)

        results[name] = {
            "fit": fit,
            "free": r_free,
            "cons": r_cons,
            "ope": ope,
            "audit": extrapolation_audit(tr.events, free, te.events),
            "by_tier": by_tier(te.events, te.oracle, cons),
            "spread_free": float(free.spreads.mean()),
            "spread_cons": float(cons.spreads.mean()),
        }

        if quiet:
            continue

        rule(f"model: {name}")
        print(f"  calibration    brier {fit['brier']:.4f}   log loss {fit['log_loss']:.4f}   "
              f"auc {fit['auc']:.4f}")
        print(f"                 ece   {fit['ece']:.4f}   mce      {fit['mce']:.4f}   "
              f"(base {fit['base_rate']:.3f} vs predicted {fit['mean_pred']:.3f})")

        print(f"\n  policy, unconstrained   mean spread {results[name]['spread_free']:6.1f} bps"
              f"   true regret {r_free['regret_target']:6.2f}"
              f"   reduction {100*r_free['regret_reduction']:6.1f}%")
        print(f"  policy, in-support      mean spread {results[name]['spread_cons']:6.1f} bps"
              f"   true regret {r_cons['regret_target']:6.2f}"
              f"   reduction {100*r_cons['regret_reduction']:6.1f}%")
        print(f"  incumbent (logged)      mean spread "
              f"{te.events.quoted_half_bps.mean():6.1f} bps"
              f"   true regret {r_cons['regret_logged']:6.2f}")

        print("\n  extrapolation audit of the unconstrained policy:")
        print("   " + results[name]["audit"].to_string(index=False, float_format=FMT)
              .replace("\n", "\n   "))

        print("\n  off-policy estimates of the in-support policy (bps per unit):")
        o = ope
        print(f"    behaviour (what happened)   {o['behaviour']:7.2f}")
        print(f"    direct  (model-on-model)    {o['direct']:7.2f}   <- circular, optimistic")
        print(f"    IPS                         {o['ips']:7.2f}   <- unbiased, high variance")
        print(f"    SNIPS                       {o['snips']:7.2f}")
        print(f"    doubly robust               {o['dr']:7.2f}")
        print(f"    truth                       {r_cons['true_margin_target']:7.2f}")
        print(f"    effective sample {o['ess']:.0f} of {len(te)} "
              f"({100*o['ess_frac']:.1f}%), max weight {o['max_weight']:.1f}")

        print("\n  per tier (in-support policy):")
        print("   " + results[name]["by_tier"].to_string(index=False, float_format=FMT)
              .replace("\n", "\n   "))

    return results


def run_seeds(n_seeds: int, weeks: int, jitter: float) -> None:
    """Regenerate data under fresh seeds and check the result is not a fluke."""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    rows = []
    for seed in range(1, n_seeds + 1):
        prefix = root / f"data/seed{seed}"
        subprocess.run(
            [str(root / "build/vane-sim"), "--weeks", str(weeks), "--seed", str(seed),
             "--jitter", str(jitter), "--out", str(prefix)],
            check=True, capture_output=True,
        )
        res = run_one(str(prefix), quiet=True)
        for name, r in res.items():
            rows.append({
                "seed": seed,
                "model": name,
                "regret_logged": r["cons"]["regret_logged"],
                "regret_free": r["free"]["regret_target"],
                "regret_cons": r["cons"]["regret_target"],
                "reduction_free": 100 * r["free"]["regret_reduction"],
                "reduction_cons": 100 * r["cons"]["regret_reduction"],
                "brier": r["fit"]["brier"],
            })

    df = pd.DataFrame(rows)
    rule(f"robustness across {n_seeds} seeds")
    print(df.to_string(index=False, float_format=FMT))
    rule("summary")
    agg = df.groupby("model").agg(
        regret_logged=("regret_logged", "mean"),
        regret_free=("regret_free", "mean"),
        regret_cons=("regret_cons", "mean"),
        reduction_free_mean=("reduction_free", "mean"),
        reduction_free_sd=("reduction_free", "std"),
        reduction_cons_mean=("reduction_cons", "mean"),
        reduction_cons_sd=("reduction_cons", "std"),
    )
    print(agg.to_string(float_format=FMT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train", help="prefix of the event/oracle CSVs")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seeds", type=int, default=0, help="run a multi-seed robustness check")
    ap.add_argument("--weeks", type=int, default=16, help="weeks per seed")
    ap.add_argument("--jitter", type=float, default=0.22)
    args = ap.parse_args()

    if args.seeds > 0:
        run_seeds(args.seeds, args.weeks, args.jitter)
    else:
        run_one(args.data, args.train_frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
