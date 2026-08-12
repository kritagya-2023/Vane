// Vane - retail FX auto-quoter
// pricer.hpp : the deterministic pricing core.
//
// Convention: rates are quote-currency per unit of base currency (INR per USD).
// `bid` is the rate at which the desk BUYS base currency from the client, `ask`
// the rate at which it SELLS. Inventory is the desk's net base-currency
// position: positive means long USD.
#pragma once

#include "vane/config.hpp"
#include "vane/types.hpp"

namespace vane {

struct QuoteRequest {
    Points   mid       = 0;  // interbank mid, in points
    Notional inventory = 0;  // desk's current net base-currency position
    Notional size      = 0;  // requested ticket size, base-currency units
    Tier     tier      = Tier::Retail;

    double sigma_daily          = 0.0;  // fractional daily vol, e.g. 0.004
    double hedge_horizon_hours  = 0.5;  // from hedge_horizon_hours(), above
    std::int64_t tick_age_ms    = 0;

    // Best competing prices, in points. Zero means "unknown".
    Points competitor_bid = 0;
    Points competitor_ask = 0;

    // Multiplicative exploration jitter on the half-spread, applied before the
    // tier and cost clamps so that deliberate randomisation can never breach a
    // commercial limit. 1.0 means no jitter. The simulator uses this to probe
    // the demand curve; Phase 3 needs the resulting variation to identify it.
    double spread_multiplier = 1.0;
};

struct Quote {
    Points bid = 0;
    Points ask = 0;
    bool   bid_valid = false;
    bool   ask_valid = false;

    // Diagnostics: every intermediate the formula produced, so that a quote can
    // be explained after the fact and so Phase 4 can regress PnL on them.
    Points reservation        = 0;
    double skew_bps           = 0.0;
    double half_spread_bps    = 0.0;
    double sigma_horizon_bps  = 0.0;
    bool   clamped_by_floor   = false;
    bool   clamped_by_ceiling = false;
    bool   clamped_by_rival   = false;

    RejectReason reject = RejectReason::None;

    bool   two_sided() const noexcept { return bid_valid && ask_valid; }
    bool   ok()        const noexcept { return bid_valid || ask_valid; }
    Points spread()    const noexcept { return ask - bid; }
};

class Pricer {
public:
    explicit Pricer(PricingConfig cfg = PricingConfig::defaults()) noexcept;

    // Pure, allocation-free, and branch-light: safe to call on a hot path.
    Quote quote(const QuoteRequest& req) const noexcept;

    const PricingConfig& config() const noexcept { return cfg_; }

private:
    PricingConfig cfg_;
};

}  // namespace vane
