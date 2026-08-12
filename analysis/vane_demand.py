"""Vane Phase 3 - the demand model.

Estimates P(accept | quoted spread, context) from logged outcomes alone.

Two models, for a reason. The logistic one is the honest baseline and, because
the simulator's true demand is a lognormal-threshold rule, a probit in
log-spread is very nearly correctly specified. Gradient boosting is the
flexible alternative that a real desk would reach for with messier data. Phase
4 compares them.

The pricing policy searches over spreads to find the revenue-maximising quote,
which means the model is evaluated at prices that were never actually shown. A
model that is merely accurate on the logged spreads is not enough: it has to be
sensibly shaped *off* them too. That is why monotonicity is enforced rather
than hoped for, and why calibration is checked rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vane_data import FEATURES, design_matrix

SPREAD_IDX = FEATURES.index("log_spread")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probabilities. Lower is better."""
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC via the Mann-Whitney statistic, ties averaged."""
    y = np.asarray(y, dtype=float)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    # average ranks within ties
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 12) -> dict:
    """Calibration curve plus expected and maximum calibration error.

    For a pricing policy this matters more than discrimination: revenue is
    `spread * P(accept)`, so a model that ranks perfectly but is 10% off in
    level will systematically misprice.
    """
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows, ece, mce, n = [], 0.0, 0.0, len(y)
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        pred, obs, cnt = float(p[m].mean()), float(y[m].mean()), int(m.sum())
        rows.append({"bin": b, "n": cnt, "predicted": pred, "observed": obs})
        gap = abs(pred - obs)
        ece += cnt / n * gap
        mce = max(mce, gap)
    return {"bins": rows, "ece": float(ece), "mce": float(mce)}


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class DemandModel:
    """Common interface: fit on logged events, predict P(accept) anywhere."""

    def fit(self, df, y, sample_weight=None):  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, df, spread=None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def accept_curve(self, row_df, spreads: np.ndarray) -> np.ndarray:
        """P(accept) across a grid of spreads for a single context row."""
        rep = row_df.iloc[[0]].loc[row_df.index.repeat(len(spreads))]
        return self.predict_proba(rep, spread=spreads)


@dataclass
class LogisticDemand(DemandModel):
    """Logistic regression in log-spread.

    Monotonicity is imposed by construction: the coefficient on log_spread is
    constrained to be negative, so raising the price can never be predicted to
    raise the chance of dealing. Without that, the policy search will happily
    walk off to whatever spread the fit happens to bend upward at.
    """

    C: float = 1.0
    scaler: StandardScaler = field(default_factory=StandardScaler)
    model: LogisticRegression | None = None
    _flipped: bool = False

    def fit(self, df, y, sample_weight=None):
        X = design_matrix(df)
        Xs = self.scaler.fit_transform(X)
        self.model = LogisticRegression(C=self.C, max_iter=2000)
        self.model.fit(Xs, y, sample_weight=sample_weight)

        coef = self.model.coef_[0][SPREAD_IDX]
        if coef > 0:
            # Demand must slope down. If the unconstrained fit disagrees the
            # data is telling us something is wrong; refuse rather than ship a
            # model the policy search will exploit.
            raise ValueError(
                f"log_spread coefficient is {coef:+.4f}: demand slopes upward, "
                "which means the fit is misspecified or the sample is degenerate"
            )
        return self

    def predict_proba(self, df, spread=None) -> np.ndarray:
        X = design_matrix(df, spread_override=spread)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


@dataclass
class BoostedDemand(DemandModel):
    """Gradient boosting with a hard monotone decreasing constraint on spread.

    scikit-learn's `monotonic_cst` enforces this inside the tree splits, so the
    guarantee survives extrapolation instead of relying on the training sample
    happening to cover the region the policy search visits.
    """

    max_iter: int = 300
    learning_rate: float = 0.06
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 40
    l2: float = 1.0
    model: HistGradientBoostingClassifier | None = None

    def fit(self, df, y, sample_weight=None):
        X = design_matrix(df)
        cst = np.zeros(X.shape[1], dtype=int)
        cst[SPREAD_IDX] = -1  # P(accept) non-increasing in log_spread
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2,
            monotonic_cst=cst,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=0,
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, df, spread=None) -> np.ndarray:
        X = design_matrix(df, spread_override=spread)
        return self.model.predict_proba(X)[:, 1]


def evaluate(model: DemandModel, df, y) -> dict:
    p = model.predict_proba(df)
    rel = reliability(y, p)
    return {
        "n": int(len(y)),
        "base_rate": float(np.mean(y)),
        "mean_pred": float(np.mean(p)),
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "auc": auc(y, p),
        "ece": rel["ece"],
        "mce": rel["mce"],
        "reliability": rel["bins"],
    }
