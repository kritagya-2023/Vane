#!/usr/bin/env python3
"""Mutation testing for Vane.

Injects a known bug into one source file, rebuilds, and asserts that at least
one test suite fails. A mutation that survives means the suites are not
actually pinning down the behaviour they claim to.

All work happens in a throwaway copy of the repository, so the working tree is
never modified and an interrupted run cannot leave an injected bug behind.

    python3 tests/mutate.py                     # everything
    python3 tests/mutate.py flow.cpp desk.cpp   # a subset
"""
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (source file, description, original snippet, mutated snippet)
MUTATIONS = [
    ("pricer.cpp", "skew sign flipped",
     "double skew = -cfg_.risk_aversion",
     "double skew = cfg_.risk_aversion"),

    ("pricer.cpp", "volatility term dropped from the spread",
     "double half = tp.base_half_bps + cfg_.vol_coeff * q.sigma_horizon_bps + size_term;",
     "double half = tp.base_half_bps + 0.0 * q.sigma_horizon_bps + size_term;"),

    ("pricer.cpp", "sqrt-of-time replaced by linear time",
     "const double sigma_h      = r.sigma_daily * std::sqrt(horizon_days);",
     "const double sigma_h      = r.sigma_daily * horizon_days;"),

    ("pricer.cpp", "economic cost floor removed",
     "if (half < cfg_.cost_floor_bps) {",
     "if (false) {"),

    ("pricer.cpp", "tier ceiling clamp removed",
     "if (half > tp.ceil_half_bps) {",
     "if (false) {"),

    ("pricer.cpp", "inventory skew cap removed",
     "skew        = std::clamp(skew, -cfg_.max_skew_bps, cfg_.max_skew_bps);",
     "skew        = skew;"),

    ("pricer.cpp", "stale tick check removed",
     "if (r.tick_age_ms > cfg_.max_tick_age_ms) {",
     "if (false) {"),

    ("pricer.cpp", "size limit check removed",
     "if (r.size > cfg_.size_max) {",
     "if (false) {"),

    ("pricer.cpp", "bid rounded up instead of down",
     "q.bid = floor_to_points(bid_rate);",
     "q.bid = ceil_to_points(bid_rate);"),

    ("pricer.cpp", "position limit sign flipped on the bid side",
     "q.bid_valid = (r.inventory + r.size) <= cfg_.inventory_hard_limit;",
     "q.bid_valid = (r.inventory - r.size) <= cfg_.inventory_hard_limit;"),

    ("pricer.cpp", "size widening applied as a discount",
     "size_term         = cfg_.size_coeff_bps * (std::sqrt(mult) - 1.0);",
     "size_term         = -cfg_.size_coeff_bps * (std::sqrt(mult) - 1.0);"),

    ("pricer.cpp", "skew applied to the spread instead of the price",
     "const double reservation = mid * (1.0 + skew / kBpsScale);",
     "const double reservation = mid;"),

    # --- Phase 2: the simulator -------------------------------------------
    ("market.cpp", "weekend gap jump removed",
     "for (int i = 0; i < reopens && cfg_.sigma_daily > 0.0; ++i) {",
     "for (int i = 0; i < 0 && cfg_.sigma_daily > 0.0; ++i) {"),

    ("market.cpp", "diffusion uses linear time instead of sqrt",
     "log_ret += -0.5 * s * s * dt + s * std::sqrt(dt) * rng_.normal();",
     "log_ret += -0.5 * s * s * dt + s * dt * rng_.normal();"),

    ("market.cpp", "market treated as open through the weekend",
     "    } else if (rem <= kSundayOpenSec) {\n        total += kFridayCloseSec;",
     "    } else if (rem <= kSundayOpenSec) {\n        total += rem;"),

    ("market.cpp", "competitor sheet never goes stale",
     "        if (refresh_in_step(j)) {",
     "        if (true) {"),

    ("flow.cpp", "price sensitivity to ticket size inverted",
     "cfg_.beta_size * std::log(static_cast<double>(c.size) /",
     "-cfg_.beta_size * std::log(static_cast<double>(c.size) /"),

    ("flow.cpp", "tier stickiness dropped from willingness to pay",
     "c.mu = std::log(c.comp_half_bps) + std::log(tp.stickiness) +",
     "c.mu = std::log(c.comp_half_bps) + 0.0 * std::log(tp.stickiness) +"),

    ("flow.cpp", "arrival thinning accepts every proposal",
     "if (rng_.uniform() * lambda_max_ <= intensity(ts)) {",
     "if (true) {"),

    ("flow.cpp", "seasonality removed from arrival intensity",
     "const double per_day = cfg_.base_arrivals_per_day * cfg_.day_factor[c.weekday] * hour_factor;",
     "const double per_day = cfg_.base_arrivals_per_day;"),

    ("flow.cpp", "oracle returns the lower bound instead of the optimum",
     "    return std::exp(0.5 * (a + b));",
     "    return std::exp(lo);"),

    ("flow.cpp", "acceptance probability ignores dispersion",
     "return norm_cdf((mu - std::log(half_bps)) / sigma_ln);",
     "return norm_cdf(mu - std::log(half_bps));"),

    ("desk.cpp", "position sign flipped when the client buys",
     "        cash_ += px * qty;\n        position_ -= size;",
     "        cash_ += px * qty;\n        position_ += size;"),

    ("desk.cpp", "hedging allowed while the market is shut",
     "    if (!is_open(t)) return 0;  // cannot hedge a closed market",
     "    if (false) return 0;"),

    ("desk.cpp", "hedge slippage not charged",
     "    const double   cost = cfg_.hedge_cost_bps / kBpsScale;",
     "    const double   cost = 0.0;"),

    ("simulator.cpp", "accept rule inverted",
     "const bool accepted = q.half_spread_bps <= c.reservation_half_bps;",
     "const bool accepted = q.half_spread_bps >= c.reservation_half_bps;"),

    ("simulator.cpp", "inventory logged after the fill instead of before",
     "        e.inventory     = desk_.position();",
     "        e.inventory     = desk_.position() + 1;"),

    ("simulator.cpp", "exploration jitter never applied",
     "req.spread_multiplier = policy_mult * mult;",
     "req.spread_multiplier = policy_mult;"),

    # --- Phase 4: the serving path ----------------------------------------
    ("policy.cpp", "support constraint ignored in the serving path",
     "    if (respect_support_) {",
     "    if (false) {"),

    ("policy.cpp", "feature order check disabled",
     "        if (feature_names[i] != kFeatureNames[i]) {",
     "        if (false && feature_names[i] != kFeatureNames[i]) {"),

    ("policy.cpp", "standardisation dropped from the linear form",
     "z += params_.coef[i] * (x[i] - params_.mean[i]) / params_.scale[i];",
     "z += params_.coef[i] * x[i];"),

    ("policy.cpp", "policy always returns the lower bound of the search",
     "    return best_d;\n}",
     "    return lo;\n}"),

    ("policy.cpp", "tier one-hot features swapped",
     "    out[10] = r.tier == Tier::Retail ? 1.0 : 0.0;",
     "    out[10] = r.tier == Tier::Wealth ? 1.0 : 0.0;"),

    ("simulator.cpp", "learned policy ignored, tier table used instead",
     "            req.spread_multiplier =\n                probe.half_spread_bps > 0.0 ? want / probe.half_spread_bps : 1.0;",
     "            req.spread_multiplier = 1.0;"),

]



SUITES = ["test_unit", "test_property", "test_sim", "test_policy"]


def main() -> int:
    # Optional substring filters, so a long run can be split up:
    #   python3 tests/mutate.py flow.cpp desk.cpp
    filters = [a for a in sys.argv[1:] if not a.startswith("-")]
    selected = [m for m in MUTATIONS
                if not filters or any(f in m[0] or f in m[1] for f in filters)]
    if not selected:
        print("no mutations matched")
        return 1

    # Everything happens inside a disposable copy of the repository. The working
    # tree is never written to, so an interrupted or killed run cannot leave an
    # injected bug behind -- which is exactly what happened when this script
    # mutated the sources in place.
    caught, survived = 0, []
    with tempfile.TemporaryDirectory(prefix="vane-mutate-") as tmp:
        work = Path(tmp) / "vane"
        shutil.copytree(ROOT, work,
                        ignore=shutil.ignore_patterns("build", ".git", "__pycache__"))

        if subprocess.run("make -s", cwd=work, shell=True,
                          capture_output=True).returncode != 0:
            print("baseline build failed in the sandbox copy")
            return 1

        originals = {f: (work / "src" / f).read_text() for f in sorted({m[0] for m in selected})}

        for fname, name, old, new in selected:
            label = f"{fname}: {name}"
            src = originals[fname]
            target = work / "src" / fname

            if old not in src:
                print(f"  SKIP      {label}  (anchor not found - update the script)")
                survived.append(label + " [anchor missing]")
                continue

            target.write_text(src.replace(old, new, 1))
            targets = " ".join(f"build/{s}" for s in SUITES)
            built = subprocess.run(f"make -s {targets}", cwd=work, shell=True,
                                   capture_output=True, text=True)
            if built.returncode != 0:
                print(f"  caught    {label}  (failed to compile)")
                caught += 1
                target.write_text(src)
                continue

            where = [s for s in SUITES
                     if subprocess.run(f"./build/{s}", cwd=work, shell=True,
                                       capture_output=True).returncode != 0]
            target.write_text(src)

            if where:
                tags = "+".join(w.replace("test_", "") for w in where)
                print(f"  caught    {label}  ({tags})")
                caught += 1
            else:
                print(f"  SURVIVED  {label}")
                survived.append(label)

    total = len(selected)
    print(f"\nmutation score: {caught}/{total} caught")
    if survived:
        print("survivors:")
        for s in survived:
            print(f"  - {s}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
