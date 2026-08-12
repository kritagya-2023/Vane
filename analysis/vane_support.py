"""Vane Phase 3 - keeping the policy inside the data.

The failure this module exists to fix is worth stating plainly, because it is
the single most instructive result in the project.

A gradient-boosted demand model, fitted with a hard monotone-decreasing
constraint on spread and well calibrated on the test set (Brier 0.178, ECE
0.011), produced a pricing policy that *destroyed* value: it quoted retail at
265 bps against a true optimum of 195, and turned a regret of 12.1 bps per unit
into 28.6.

The cause is not the fit. It is that trees cannot extrapolate. Beyond the
largest logged spread the model's prediction is whatever the final split said,
held flat forever:

    spread   true P(accept)   boosted
       320       0.362         0.409
       380       0.243         0.301
       450       0.150         0.301
       550       0.076         0.301

Revenue is `(spread - cost) * P(accept)`. Against a flat tail that is a
straight line in spread, so the argmax runs to the edge of the grid. The
monotone constraint stopped the curve from rising; it did nothing about flat,
and flat is enough.

The remedy is not a better model. No amount of fitting recovers information the
data does not contain. The remedy is to stop the policy from making decisions
in regions where the logged data cannot support one, and to say so explicitly
rather than quietly clipping.

Two mechanisms:

  support bounds - restrict the search to a per-tier quantile range of the
                   logged spreads, so the policy only ever proposes prices the
                   data speaks to.
  extrapolation  - report how far each chosen spread sits outside the logged
      audit        support, so silent extrapolation is visible instead of
                   being absorbed into an average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vane_policy import COST_FLOOR_BPS, TIER_BANDS, PolicyResult, clamp_to_band


def support_bounds(
    train_df: pd.DataFrame, lo_q: float = 0.02, hi_q: float = 0.98
) -> dict[str, tuple[float, float]]:
    """Per-tier spread range the logged data actually covers.

    Quantiles rather than min/max, because the extremes are single events and a
    policy should not be steered by one observation.
    """
    out = {}
    for tier, grp in train_df.groupby("tier"):
        s = grp["quoted_half_bps"]
        band_lo, band_hi = TIER_BANDS[tier]
        out[tier] = (
            max(float(s.quantile(lo_q)), band_lo, COST_FLOOR_BPS),
            min(float(s.quantile(hi_q)), band_hi),
        )
    return out


def optimal_spread_in_support(
    model,
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    cost_bps: float = 1.5,
    grid: int = 96,
    refine: int = 3,
) -> PolicyResult:
    """Revenue-maximising quote, searched only where the data has support."""
    lo = df["tier"].map(lambda t: bounds[t][0]).to_numpy(dtype=float)
    hi = df["tier"].map(lambda t: bounds[t][1]).to_numpy(dtype=float)
    hi = np.maximum(hi, lo * 1.001)

    best_d = lo.copy()
    best_v = np.full(len(df), -np.inf)

    def scan(lo_, hi_, k):
        nonlocal best_d, best_v
        for i in range(k):
            frac = i / (k - 1)
            d = np.exp(np.log(lo_) + frac * (np.log(hi_) - np.log(lo_)))
            v = (d - cost_bps) * model.predict_proba(df, spread=d)
            better = v > best_v
            best_v = np.where(better, v, best_v)
            best_d = np.where(better, d, best_d)

    scan(lo, hi, grid)
    width = (np.log(hi) - np.log(lo)) / (grid - 1)
    for _ in range(refine):
        scan(np.maximum(np.exp(np.log(best_d) - width), lo),
             np.minimum(np.exp(np.log(best_d) + width), hi), 12)
        width /= 4.0

    spreads = clamp_to_band(best_d, df["tier"], cost_bps)
    p = model.predict_proba(df, spread=spreads)
    return PolicyResult(spreads, p, (spreads - cost_bps) * p)


def extrapolation_audit(
    train_df: pd.DataFrame, target: PolicyResult, df: pd.DataFrame
) -> pd.DataFrame:
    """How often, and how far, the policy prices outside the logged support.

    A policy that looks good on average while quoting 30% of its flow outside
    the data is not a good policy; it is an untested one. This makes that
    visible.
    """
    rows = []
    for tier, grp in train_df.groupby("tier"):
        m = (df["tier"] == tier).to_numpy()
        if not m.any():
            continue
        lo, hi = float(grp["quoted_half_bps"].quantile(0.02)), float(
            grp["quoted_half_bps"].quantile(0.98)
        )
        s = target.spreads[m]
        above = s > hi
        rows.append({
            "tier": tier,
            "n": int(m.sum()),
            "support_lo": lo,
            "support_hi": hi,
            "mean_spread": float(s.mean()),
            "pct_above": 100.0 * float(above.mean()),
            "pct_below": 100.0 * float((s < lo).mean()),
            "worst_overshoot": float((s[above] / hi).max() - 1.0) * 100.0 if above.any() else 0.0,
        })
    return pd.DataFrame(rows)
