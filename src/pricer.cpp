#include "vane/pricer.hpp"

#include <algorithm>
#include <cmath>

namespace vane {

Pricer::Pricer(PricingConfig cfg) noexcept : cfg_(cfg) {}

Quote Pricer::quote(const QuoteRequest& r) const noexcept {
    Quote q;

    // -- 1. validation and guardrails ---------------------------------------
    if (r.mid < cfg_.min_mid || r.size < 0 || r.sigma_daily < 0.0 || r.hedge_horizon_hours < 0.0 ||
        !std::isfinite(r.sigma_daily) || !std::isfinite(r.hedge_horizon_hours)) {
        q.reject = RejectReason::InvalidInput;
        return q;
    }
    if (r.tick_age_ms > cfg_.max_tick_age_ms) {
        q.reject = RejectReason::StaleTick;
        return q;
    }
    if (r.size > cfg_.size_max) {
        q.reject = RejectReason::SizeAboveLimit;
        return q;
    }

    const TierParams& tp  = cfg_.tiers[static_cast<int>(r.tier)];
    const double      mid = to_rate(r.mid);

    // -- 2. volatility over the hedging horizon ------------------------------
    // sigma scales with the square root of time, so a 48-hour weekend gap is
    // roughly 10x the risk of a half-hour intraday hold, not 96x.
    const double horizon_days = r.hedge_horizon_hours / 24.0;
    const double sigma_h      = r.sigma_daily * std::sqrt(horizon_days);
    q.sigma_horizon_bps       = sigma_h * kBpsScale;

    // -- 3. reservation price (Avellaneda-Stoikov inventory skew) ------------
    // Long base currency -> shade the whole two-way price down, so the ask is
    // more attractive and the bid less so. The desk gets lifted out of its
    // position instead of accumulating more of it.
    const double q_norm =
        static_cast<double>(r.inventory) / static_cast<double>(cfg_.inventory_limit);
    double skew = -cfg_.risk_aversion * tp.skew_mult * q_norm * q.sigma_horizon_bps;
    skew        = std::clamp(skew, -cfg_.max_skew_bps, cfg_.max_skew_bps);
    q.skew_bps  = skew;

    const double reservation = mid * (1.0 + skew / kBpsScale);
    q.reservation            = round_to_points(reservation);

    // -- 4. half-spread ------------------------------------------------------
    // Larger tickets are harder to work out of, so they widen with the square
    // root of size. Tier discounts are expressed through base/floor/ceil, not
    // here. Phase 3 replaces `base_half_bps` with the revenue-maximising
    // spread implied by the fitted demand curve.
    double size_term = 0.0;
    if (r.size > cfg_.size_ref) {
        const double mult = static_cast<double>(r.size) / static_cast<double>(cfg_.size_ref);
        size_term         = cfg_.size_coeff_bps * (std::sqrt(mult) - 1.0);
    }

    double half = tp.base_half_bps + cfg_.vol_coeff * q.sigma_horizon_bps + size_term;

    // Exploration jitter, before the clamps: randomising the quote must never
    // be able to push it outside the approved band.
    if (r.spread_multiplier > 0.0 && std::isfinite(r.spread_multiplier)) {
        half *= r.spread_multiplier;
    }

    if (half > tp.ceil_half_bps) {
        half                  = tp.ceil_half_bps;
        q.clamped_by_ceiling  = true;
    }
    if (half < tp.floor_half_bps) {
        half               = tp.floor_half_bps;
        q.clamped_by_floor = true;
    }
    // The economic floor covers funding, vaulting and operations. It is
    // absolute: no tier discount and no competitor may price through it.
    if (half < cfg_.cost_floor_bps) {
        half               = cfg_.cost_floor_bps;
        q.clamped_by_floor = true;
    }
    q.half_spread_bps = half;

    double bid_rate = reservation * (1.0 - half / kBpsScale);
    double ask_rate = reservation * (1.0 + half / kBpsScale);

    // -- 5. optional competitor clamp ---------------------------------------
    if (cfg_.competitive_clamp_enabled) {
        const double slip      = cfg_.competitive_max_slip_bps / kBpsScale;
        const double floor_off = cfg_.cost_floor_bps / kBpsScale;
        const double bid_cap   = reservation * (1.0 - floor_off);
        const double ask_cap   = reservation * (1.0 + floor_off);

        if (r.competitor_bid > 0) {
            const double target = to_rate(r.competitor_bid) * (1.0 - slip);
            if (target > bid_rate) {
                bid_rate          = std::min(target, bid_cap);
                q.clamped_by_rival = true;
            }
        }
        if (r.competitor_ask > 0) {
            const double target = to_rate(r.competitor_ask) * (1.0 + slip);
            if (target < ask_rate) {
                ask_rate          = std::max(target, ask_cap);
                q.clamped_by_rival = true;
            }
        }
    }

    // -- 6. fixed-point conversion, rounded in the desk's favour -------------
    q.bid = floor_to_points(bid_rate);
    q.ask = ceil_to_points(ask_rate);

    // -- 7. position limits may leave the quote one-sided --------------------
    // Buying at the bid grows the position; selling at the ask shrinks it. At
    // the hard limit the desk stops showing the side that would make things
    // worse but keeps showing the side that unwinds.
    q.bid_valid = (r.inventory + r.size) <= cfg_.inventory_hard_limit;
    q.ask_valid = (r.inventory - r.size) >= -cfg_.inventory_hard_limit;

    if (!q.bid_valid && !q.ask_valid) {
        q.reject = RejectReason::InventoryHardLimit;
    }
    return q;
}

}  // namespace vane
