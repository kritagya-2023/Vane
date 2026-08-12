"""Vane Phase 3 - data loading and feature construction.

The event log is what a desk could observe at quote time. The oracle log holds
the hidden truth and is loaded only for scoring, never for fitting: `load()`
returns them separately and every training path in this package takes the
former alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIERS = ["retail", "corporate", "private", "wealth"]
SIZE_REF = 5_000.0

# Columns a model is allowed to see. Anything outside this list is either the
# hidden truth or a leak from the label.
FEATURES = [
    "log_spread",       # the decision variable
    "log_comp",         # the competitor's effective half-spread
    "log_size",         # ticket size relative to the reference
    "log_horizon",      # hedging horizon, hours
    "sigma_daily",
    "abs_inventory",    # desk pressure, normalised
    "is_buy",
    "hour_sin",
    "hour_cos",
    "is_weekend_flow",  # arrived while the interbank market was shut
] + [f"tier_{t}" for t in TIERS]


@dataclass
class Dataset:
    events: pd.DataFrame
    oracle: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self.events)


def load(prefix: str | Path, with_oracle: bool = True) -> Dataset:
    """Load an events CSV, and optionally its oracle companion.

    The oracle frame is aligned on `id` and returned separately so that a
    training routine that only ever receives `events` cannot see it.
    """
    prefix = Path(prefix)
    events = pd.read_csv(f"{prefix}_events.csv")
    events = add_features(events)

    oracle = None
    if with_oracle:
        opath = Path(f"{prefix}_oracle.csv")
        if opath.exists():
            oracle = pd.read_csv(opath)
            if not (oracle["id"].to_numpy() == events["id"].to_numpy()).all():
                raise ValueError("event and oracle logs are not aligned on id")
    return Dataset(events=events, oracle=oracle)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from the raw event log.

    Everything here is a transform of columns the desk observes before the
    client responds. `accepted` is never used.
    """
    df = df.copy()

    df["log_spread"] = np.log(df["quoted_half_bps"])
    df["log_comp"] = np.log(df["comp_half_bps"])
    df["log_size"] = np.log(df["size"] / SIZE_REF)
    # horizon can be zero at the very start of a step; nudge it off the floor
    df["log_horizon"] = np.log(np.maximum(df["horizon_h"], 0.05))
    df["abs_inventory"] = np.abs(df["inventory"]) / 1e6
    df["is_buy"] = df["client_buys"].astype(float)
    df["is_weekend_flow"] = (1.0 - df["market_open"]).astype(float)

    # Hour of day as a circle, so 23:00 and 01:00 are neighbours.
    hour = df["hour_utc"].to_numpy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    for t in TIERS:
        df[f"tier_{t}"] = (df["tier"] == t).astype(float)

    # Propensity of the logged action under the behaviour policy. The jitter is
    # log-normal with zero mean in log space, so the density of the realised
    # multiplier is what reweights the sample in Phase 3's IPS estimator.
    sd = df["jitter_log_sd"].to_numpy()
    z = np.log(df["spread_mult"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(
            sd > 0,
            np.exp(-0.5 * (z / np.where(sd > 0, sd, 1.0)) ** 2)
            / (np.where(sd > 0, sd, 1.0) * np.sqrt(2 * np.pi)),
            1.0,
        )
    df["log_propensity"] = np.log(np.maximum(dens, 1e-12))
    return df


def time_split(ds: Dataset, train_frac: float = 0.7) -> tuple[Dataset, Dataset]:
    """Split chronologically, never at random.

    A random split would let the model train on Thursday and test on Wednesday
    of the same week, with the same price level and the same competitor sheet
    on both sides. That flatters the model and tells you nothing about whether
    it would have worked live.
    """
    ev = ds.events.sort_values("t_sec").reset_index(drop=True)
    cut_t = ev["t_sec"].quantile(train_frac)
    tr_mask = ev["t_sec"] <= cut_t

    def take(mask: pd.Series) -> Dataset:
        e = ev[mask].reset_index(drop=True)
        o = None
        if ds.oracle is not None:
            o = ds.oracle.set_index("id").loc[e["id"]].reset_index()
        return Dataset(events=e, oracle=o)

    return take(tr_mask), take(~tr_mask)


def design_matrix(df: pd.DataFrame, spread_override: np.ndarray | None = None) -> np.ndarray:
    """Build the feature matrix, optionally at a counterfactual spread.

    `spread_override` is what makes the policy search possible: it asks the
    fitted model what would have happened at prices that were never quoted.
    """
    cols = []
    for name in FEATURES:
        if name == "log_spread" and spread_override is not None:
            cols.append(np.log(spread_override))
        else:
            cols.append(df[name].to_numpy(dtype=float))
    return np.column_stack(cols)
