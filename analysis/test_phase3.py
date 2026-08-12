#!/usr/bin/env python3
"""Tests for the Phase 3 analysis layer.

Two kinds of check. Ordinary unit tests on the estimators and transforms, and
statistical tests that generate data with a known answer and confirm the code
recovers it. The latter matter more here: a demand model that silently fits the
wrong thing still returns plausible numbers.

    python3 analysis/test_phase3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vane_data import FEATURES, add_features, design_matrix, load, time_split  # noqa: E402
from vane_demand import (  # noqa: E402
    BoostedDemand,
    LogisticDemand,
    auc,
    brier,
    log_loss,
    reliability,
)
from vane_policy import (  # noqa: E402
    TIER_BANDS,
    PolicyResult,
    clamp_to_band,
    off_policy_value,
    optimal_spread,
    oracle_regret,
)
from vane_support import (  # noqa: E402
    extrapolation_audit,
    optimal_spread_in_support,
    support_bounds,
)

CHECKS = {"n": 0, "failed": 0}


def check(cond: bool, msg: str) -> None:
    CHECKS["n"] += 1
    if not cond:
        CHECKS["failed"] += 1
        print(f"  FAIL  {msg}")


def section(name: str) -> None:
    print(f"\n-- {name}")


# ---------------------------------------------------------------------------
def synth(n: int = 20_000, seed: int = 0, jitter: float = 0.25,
          base: float = 200.0, mu_mult: float = 1.3, sigma_ln: float = 0.45) -> pd.DataFrame:
    """Synthetic log with a demand curve whose parameters we choose."""
    rng = np.random.default_rng(seed)
    comp = np.full(n, 130.0)
    mult = np.exp(jitter * rng.standard_normal(n))
    quoted = base * mult
    mu = np.log(comp * mu_mult)
    reservation = np.exp(mu + sigma_ln * rng.standard_normal(n))

    df = pd.DataFrame({
        "id": np.arange(n),
        "t_sec": np.sort(rng.integers(0, 86_400 * 60, n)),
        "weekday": rng.integers(0, 7, n),
        "hour_utc": rng.uniform(0, 24, n),
        "market_open": 1,
        "tier": "retail",
        "client_buys": rng.integers(0, 2, n),
        "size": 5_000,
        "mid": 95.29,
        "inventory": 0,
        "sigma_daily": 0.004,
        "horizon_h": 0.5,
        "comp_half_bps": comp,
        "base_half_bps": base,
        "spread_mult": mult,
        "jitter_log_sd": jitter,
        "quoted_half_bps": quoted,
        "clamped": 0,
        "quote_px": 95.29,
        "accepted": (quoted <= reservation).astype(int),
    })
    df = add_features(df)
    df.attrs["mu"] = mu
    df.attrs["sigma_ln"] = sigma_ln
    return df


# ---------------------------------------------------------------------------
def test_metrics() -> None:
    section("metrics")

    y = np.array([0, 0, 1, 1.0])
    check(abs(brier(y, y) - 0.0) < 1e-12, "brier of a perfect forecast is zero")
    check(abs(brier(y, np.full(4, 0.5)) - 0.25) < 1e-12, "brier of a coin flip is 0.25")
    check(log_loss(y, y) < 1e-6, "log loss of a perfect forecast is ~0")
    check(abs(log_loss(y, np.full(4, 0.5)) - np.log(2)) < 1e-9, "log loss of 0.5 is ln 2")

    check(abs(auc(y, np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-12, "auc of perfect ranking")
    check(abs(auc(y, np.array([0.9, 0.8, 0.2, 0.1])) - 0.0) < 1e-12, "auc of reversed ranking")
    check(abs(auc(y, np.full(4, 0.5)) - 0.5) < 1e-12, "auc with all ties is 0.5")
    check(np.isnan(auc(np.ones(4), np.arange(4.0))), "auc undefined with one class")

    # A well-calibrated forecast has near-zero ECE; a shifted one does not.
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 40_000)
    yy = (rng.uniform(size=40_000) < p).astype(float)
    check(reliability(yy, p)["ece"] < 0.01, "calibrated forecast has small ECE")
    check(reliability(yy, np.clip(p + 0.15, 0, 1))["ece"] > 0.10,
          "a 15-point shift shows up as large ECE")


def test_features() -> None:
    section("features and splitting")

    df = synth(2_000)
    check(all(f in df.columns for f in FEATURES), "every declared feature is built")
    check("accepted" not in FEATURES, "the label is not a feature")
    check("reservation_half_bps" not in df.columns, "hidden truth is not in the event frame")

    X = design_matrix(df)
    check(X.shape == (len(df), len(FEATURES)), "design matrix has the right shape")
    check(np.isfinite(X).all(), "design matrix is finite")

    # A spread override must change only the spread column.
    X2 = design_matrix(df, spread_override=np.full(len(df), 250.0))
    idx = FEATURES.index("log_spread")
    check(np.allclose(X2[:, idx], np.log(250.0)), "override sets the spread column")
    other = [i for i in range(X.shape[1]) if i != idx]
    check(np.allclose(X[:, other], X2[:, other]), "override leaves other columns alone")

    # Propensity: the density of the logged multiplier under the jitter.
    expected = norm.logpdf(np.log(df["spread_mult"]), 0.0, df["jitter_log_sd"])
    check(np.allclose(df["log_propensity"], expected), "log propensity matches the jitter density")

    # Time split must not interleave.
    from vane_data import Dataset

    tr, te = time_split(Dataset(events=df), 0.7)
    check(tr.events["t_sec"].max() <= te.events["t_sec"].min(),
          "train precedes test with no overlap")
    check(len(tr) + len(te) == len(df), "split preserves every row")


def test_demand_recovery() -> None:
    section("demand model recovers a known curve")

    df = synth(30_000, seed=2)
    y = df["accepted"].to_numpy()
    mu, s = df.attrs["mu"][0], df.attrs["sigma_ln"]

    for name, model in [("logistic", LogisticDemand()), ("boosted", BoostedDemand())]:
        m = model.fit(df, y)
        grid = np.array([120.0, 160.0, 200.0, 240.0, 280.0])
        rep = df.iloc[[0]].loc[df.index[[0]].repeat(len(grid))]
        pred = m.predict_proba(rep, spread=grid)
        true = norm.cdf((mu - np.log(grid)) / s)
        err = np.max(np.abs(pred - true))
        check(err < 0.06, f"{name} recovers P(accept) inside support (max err {err:.3f})")
        # Monotone: raising the price never raises the chance of dealing.
        check(np.all(np.diff(pred) <= 1e-9), f"{name} demand is non-increasing in spread")

    # An upward-sloping sample must be refused, not fitted.
    bad = synth(5_000, seed=3)
    bad["accepted"] = (bad["quoted_half_bps"] > bad["quoted_half_bps"].median()).astype(int)
    try:
        LogisticDemand().fit(bad, bad["accepted"].to_numpy())
        check(False, "upward-sloping demand should be refused")
    except ValueError:
        check(True, "upward-sloping demand is refused")


def test_policy() -> None:
    section("pricing policy")

    df = synth(20_000, seed=4)
    y = df["accepted"].to_numpy()
    m = LogisticDemand().fit(df, y)
    pol = optimal_spread(m, df, cost_bps=1.5)

    check(len(pol.spreads) == len(df), "one spread per event")
    check(np.all(pol.spreads > 1.5), "every quote clears the cost")
    check(np.all(np.isfinite(pol.expected_margin)), "expected margin is finite")

    # The chosen spread must beat a dense grid of alternatives under the model.
    sub = df.iloc[:400]
    sub_pol = optimal_spread(m, sub, cost_bps=1.5)
    # Tolerance is relative: the objective is order 100 bps, so an absolute
    # 1e-6 threshold sits below double-precision noise in the model call.
    worst = 0.0
    for d in np.exp(np.linspace(np.log(20), np.log(500), 120)):
        v = (d - 1.5) * m.predict_proba(sub, spread=np.full(len(sub), d))
        worst = max(worst, float(np.max(v - sub_pol.expected_margin)))
    rel = worst / float(np.mean(sub_pol.expected_margin))
    check(rel < 1e-4, f"grid beat the chosen spread by {rel:.2e} relative")

    # Against the true curve, the optimum should sit near the analytic one.
    mu, s = df.attrs["mu"][0], df.attrs["sigma_ln"]
    grid = np.exp(np.linspace(np.log(20), np.log(600), 4000))
    true_v = (grid - 1.5) * norm.cdf((mu - np.log(grid)) / s)
    true_best = grid[np.argmax(true_v)]
    check(abs(pol.spreads.mean() / true_best - 1.0) < 0.10,
          f"policy {pol.spreads.mean():.1f} near true optimum {true_best:.1f}")

    # Band clamping respects the commercial limits.
    tiers = pd.Series(["retail", "wealth", "corporate", "private"])
    clamped = clamp_to_band(np.array([5.0, 5.0, 9999.0, 9999.0]), tiers, 1.5)
    check(clamped[0] >= TIER_BANDS["retail"][0], "clamp lifts to the tier floor")
    check(clamped[2] <= TIER_BANDS["corporate"][1], "clamp caps at the tier ceiling")
    check(np.all(clamped >= 12.0), "clamp never prices below the cost floor")


def test_support() -> None:
    section("support bounds and extrapolation audit")

    df = synth(20_000, seed=5)
    b = support_bounds(df)
    check("retail" in b, "bounds computed per tier")
    lo, hi = b["retail"]
    check(lo < df["quoted_half_bps"].median() < hi, "bounds bracket the median")
    check(lo >= 12.0, "bounds respect the cost floor")
    check(hi <= TIER_BANDS["retail"][1], "bounds respect the tier ceiling")

    m = LogisticDemand().fit(df, df["accepted"].to_numpy())
    cons = optimal_spread_in_support(m, df, b)
    check(np.all(cons.spreads >= lo - 1e-9) and np.all(cons.spreads <= hi + 1e-9),
          "constrained policy stays inside the support")

    # The audit must flag a policy that deliberately prices out of support.
    wild = PolicyResult(np.full(len(df), hi * 2.0), np.zeros(len(df)), np.zeros(len(df)))
    a = extrapolation_audit(df, wild, df)
    check(float(a.loc[a.tier == "retail", "pct_above"].iloc[0]) > 99.0,
          "audit flags an out-of-support policy")
    inside = extrapolation_audit(df, cons, df)
    check(float(inside.loc[inside.tier == "retail", "pct_above"].iloc[0]) < 1.0,
          "audit is quiet for an in-support policy")


def test_off_policy() -> None:
    section("off-policy estimators")

    df = synth(30_000, seed=6)
    y = df["accepted"].to_numpy()
    m = LogisticDemand().fit(df, y)

    # The identity that must hold exactly: a target which *is* the behaviour
    # policy has importance weights of one everywhere, so every estimator
    # returns the realised value. This is the check that caught the original
    # coordinate bug in the importance weight, where the target's density in
    # log-spread was compared against the behaviour density in log-multiplier.
    same = PolicyResult(df["base_half_bps"].to_numpy(), np.zeros(len(df)), np.zeros(len(df)))
    r = off_policy_value(df, same, model=m)
    check(abs(r["snips"] / r["behaviour"] - 1.0) < 1e-6,
          f"SNIPS exactly recovers behaviour value ({r['snips']:.4f} vs {r['behaviour']:.4f})")
    check(abs(r["ips"] / r["behaviour"] - 1.0) < 1e-6, "IPS exactly recovers behaviour value")
    check(abs(r["mean_weight"] - 1.0) < 1e-9, "identical policies give unit weights")
    check(r["ess_frac"] > 0.999, "identical policies give a full effective sample")

    # A far-away target must show up as a collapsed effective sample.
    far = PolicyResult(np.full(len(df), 500.0), np.zeros(len(df)), np.zeros(len(df)))
    rf = off_policy_value(df, far, model=m)
    check(rf["ess_frac"] < r["ess_frac"],
          "effective sample falls as the target drifts from the data")

    # Weights are clipped, so no single event can dominate.
    check(rf["max_weight"] <= 20.0 + 1e-9, "importance weights are clipped")


def test_oracle_scoring() -> None:
    section("oracle scoring")

    df = synth(5_000, seed=7)
    mu, s = df.attrs["mu"], df.attrs["sigma_ln"]
    grid = np.exp(np.linspace(np.log(20), np.log(600), 3000))
    best = np.max((grid - 1.5) * norm.cdf((mu[0] - np.log(grid)) / s))

    oracle = pd.DataFrame({
        "id": df["id"],
        "mu": mu,
        "sigma_ln": s,
        "oracle_margin_bps": best,
        "oracle_half_bps": grid[np.argmax((grid - 1.5) * norm.cdf((mu[0] - np.log(grid)) / s))],
    })

    at_best = PolicyResult(np.full(len(df), oracle["oracle_half_bps"].iloc[0]),
                           np.zeros(len(df)), np.zeros(len(df)))
    r = oracle_regret(df, oracle, at_best)
    check(abs(r["regret_target"]) < 1e-6, "quoting the true optimum has zero regret")
    check(r["regret_logged"] > 0, "the logged policy has positive regret")

    worse = PolicyResult(np.full(len(df), 450.0), np.zeros(len(df)), np.zeros(len(df)))
    rw = oracle_regret(df, oracle, worse)
    check(rw["regret_target"] > r["regret_target"], "a worse price has larger regret")


def test_extrapolation_failure() -> None:
    section("the extrapolation failure is real and is fixed")

    # The central finding of Phase 3, pinned by a test so a future change
    # cannot quietly un-find it: a tree model asked to price beyond its data
    # predicts a flat tail, the policy walks out to meet it, and value is
    # destroyed. Support constraints fix it.
    #
    # This needs the real simulator log, not the synthetic one. The synthetic
    # frame holds a single repeated context, so its "support" is just the
    # jitter width and the failure never triggers. Skipping when the data is
    # absent is deliberate: a test that silently passes on the wrong data is
    # worse than one that says it did not run.
    prefix = Path(__file__).resolve().parent.parent / "data/train"
    if not Path(f"{prefix}_events.csv").exists():
        print("  SKIP  data/train_events.csv not found; run vane-sim --out data/train")
        return

    ds = load(str(prefix))
    tr, te = time_split(ds, 0.7)
    ytr = tr.events["accepted"].to_numpy()
    b = support_bounds(tr.events)
    hi = b["retail"][1]

    bs = BoostedDemand().fit(tr.events, ytr)
    row = te.events[te.events["tier"] == "retail"].iloc[[0]]
    grid = np.array([hi * 1.15, hi * 1.5, hi * 2.0])
    rep = row.loc[row.index.repeat(len(grid))]
    tail = bs.predict_proba(rep, spread=grid)
    check(float(np.ptp(tail)) < 0.02, "boosted prediction is flat beyond the logged support")

    lg = LogisticDemand().fit(tr.events, ytr)
    tail_lg = lg.predict_proba(rep, spread=grid)
    check(float(np.ptp(tail_lg)) > float(np.ptp(tail)),
          "the parametric model keeps falling where the tree flatlines")

    free = optimal_spread(bs, te.events)
    cons = optimal_spread_in_support(bs, te.events, b)
    check(free.spreads.mean() > cons.spreads.mean() + 20.0,
          f"unconstrained quotes far wider ({free.spreads.mean():.0f} vs "
          f"{cons.spreads.mean():.0f} bps)")
    check(np.all(cons.spreads <= hi + 1e-9), "the constrained policy stays in support")

    # And the damage is real when scored against the truth.
    r_free = oracle_regret(te.events, te.oracle, free)
    r_cons = oracle_regret(te.events, te.oracle, cons)
    check(r_free["regret_target"] > r_cons["regret_target"],
          "the unconstrained policy has worse true regret")
    check(r_free["regret_target"] > r_free["regret_logged"],
          "unconstrained boosting is worse than the incumbent it replaced")
    check(r_cons["regret_target"] < r_cons["regret_logged"],
          "the in-support policy genuinely beats the incumbent")


def main() -> int:
    test_metrics()
    test_features()
    test_demand_recovery()
    test_policy()
    test_support()
    test_off_policy()
    test_oracle_scoring()
    test_extrapolation_failure()
    print(f"\nphase 3: {CHECKS['n']} checks, {CHECKS['failed']} failed")
    return 1 if CHECKS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
