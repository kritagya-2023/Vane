// Vane - retail FX auto-quoter
// config.hpp : all tunable pricing parameters in one struct.
#pragma once

#include <string>

#include "vane/types.hpp"

namespace vane {

// Per-segment markup band. `base_half_bps` is the starting half-spread before
// any volatility or size adjustment; floor/ceil bound the result so that no
// input can push a tier outside its commercially approved band.
struct TierParams {
    double base_half_bps  = 0.0;
    double floor_half_bps = 0.0;
    double ceil_half_bps  = 0.0;
    double skew_mult      = 1.0;  // how hard inventory pressure moves this tier
};

struct PricingConfig {
    TierParams tiers[kTierCount];

    // --- inventory skew -----------------------------------------------------
    double   risk_aversion   = 1.0;      // dimensionless gamma
    Notional inventory_limit = 2'000'000;   // normalising scale for q
    Notional inventory_hard_limit = 5'000'000;  // beyond this, quote one side
    double   max_skew_bps    = 120.0;    // cap on price displacement

    // --- spread -------------------------------------------------------------
    double   vol_coeff      = 0.75;   // horizon vol -> half-spread transfer
    double   size_coeff_bps = 25.0;   // widening per sqrt-multiple of size_ref
    Notional size_ref       = 5'000;
    Notional size_max       = 250'000;   // FEMA LRS annual cap, in USD
    double   cost_floor_bps = 12.0;   // funding + vaulting + ops; absolute floor

    // --- guardrails ---------------------------------------------------------
    std::int64_t max_tick_age_ms = 2'000;
    // A mid this small has no fixed-point resolution left to carry a spread:
    // the bid would truncate to zero. 0.01 sits well below any real pair
    // (JPY/INR is around 0.6), so this only ever catches corrupt input.
    Points min_mid = 1'000;

    // --- optional competitor clamp -----------------------------------------
    // Off by default. A crude stand-in for demand response that Phase 3
    // replaces with a calibrated P(accept) curve; kept so Phase 4 can ablate
    // "heuristic competitiveness" against "learned competitiveness".
    bool   competitive_clamp_enabled = false;
    double competitive_max_slip_bps  = 30.0;

    static PricingConfig defaults() noexcept;

    // Returns empty string when the config is internally consistent, otherwise
    // a human-readable description of the first problem found.
    std::string validate() const;
};

}  // namespace vane
