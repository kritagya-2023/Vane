#include <cmath>
#include <sstream>

#include "vane/calendar.hpp"
#include "vane/config.hpp"
#include "vane/types.hpp"

namespace vane {

// ---------------------------------------------------------------------------
// types
// ---------------------------------------------------------------------------
std::string_view tier_name(Tier t) noexcept {
    switch (t) {
        case Tier::Retail:    return "retail";
        case Tier::Corporate: return "corporate";
        case Tier::Private:   return "private";
        case Tier::Wealth:    return "wealth";
    }
    return "unknown";
}

bool parse_tier(std::string_view s, Tier& out) noexcept {
    if (s == "retail")    { out = Tier::Retail;    return true; }
    if (s == "corporate") { out = Tier::Corporate; return true; }
    if (s == "private")   { out = Tier::Private;   return true; }
    if (s == "wealth")    { out = Tier::Wealth;    return true; }
    return false;
}

std::string_view reject_name(RejectReason r) noexcept {
    switch (r) {
        case RejectReason::None:               return "none";
        case RejectReason::InvalidInput:       return "invalid_input";
        case RejectReason::StaleTick:          return "stale_tick";
        case RejectReason::SizeAboveLimit:     return "size_above_limit";
        case RejectReason::InventoryHardLimit: return "inventory_hard_limit";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------
// Defaults are anchored to the rate snapshot in the internship appendix
// (interbank USD/INR 95.29, 6 Jul 2026):
//
//   ICICI       buy margin 1.98 INR = 208 bps | sell 1.84 = 193 bps
//   HDFC        buy        2.50     = 262 bps | sell 2.74 = 288 bps
//   Kotak       buy        2.74     = 288 bps | sell 1.62 = 170 bps
//   Thomas Cook buy        1.50     = 157 bps | sell 0.99 = 104 bps
//
// So a ~200 bps retail half-spread reproduces incumbent bank pricing, and the
// specialist sits near 100-160 bps. That gap is exactly what the learned
// quoter in Phase 3 has to close without giving away the margin.
PricingConfig PricingConfig::defaults() noexcept {
    PricingConfig c;
    //                      base   floor   ceil   skew_mult
    c.tiers[0] = TierParams{200.0,  40.0, 400.0, 1.00};  // Retail
    c.tiers[1] = TierParams{ 70.0,  20.0, 220.0, 1.00};  // Corporate
    c.tiers[2] = TierParams{ 40.0,  15.0, 140.0, 0.85};  // Private
    c.tiers[3] = TierParams{ 25.0,  12.0,  90.0, 0.70};  // Wealth
    return c;
}

std::string PricingConfig::validate() const {
    std::ostringstream err;
    for (int i = 0; i < kTierCount; ++i) {
        const TierParams& t  = tiers[i];
        std::string_view  nm = tier_name(static_cast<Tier>(i));
        if (t.floor_half_bps <= 0.0) {
            err << "tier " << nm << ": floor_half_bps must be positive";
            return err.str();
        }
        if (t.ceil_half_bps < t.floor_half_bps) {
            err << "tier " << nm << ": ceil_half_bps below floor_half_bps";
            return err.str();
        }
        if (t.base_half_bps < t.floor_half_bps || t.base_half_bps > t.ceil_half_bps) {
            err << "tier " << nm << ": base_half_bps outside [floor, ceil]";
            return err.str();
        }
        if (t.skew_mult < 0.0) {
            err << "tier " << nm << ": skew_mult must be non-negative";
            return err.str();
        }
        // The economic floor is absolute and is applied after the tier clamp,
        // so a ceiling below it would be silently unreachable.
        if (cost_floor_bps > t.ceil_half_bps) {
            err << "tier " << nm << ": cost_floor_bps exceeds ceil_half_bps";
            return err.str();
        }
    }
    if (risk_aversion < 0.0)        return "risk_aversion must be non-negative";
    if (vol_coeff < 0.0)            return "vol_coeff must be non-negative";
    if (size_coeff_bps < 0.0)       return "size_coeff_bps must be non-negative";
    if (cost_floor_bps <= 0.0)      return "cost_floor_bps must be positive";
    if (max_skew_bps < 0.0)         return "max_skew_bps must be non-negative";
    if (inventory_limit <= 0)       return "inventory_limit must be positive";
    if (inventory_hard_limit <= 0)  return "inventory_hard_limit must be positive";
    if (size_ref <= 0)              return "size_ref must be positive";
    if (size_max <= 0)              return "size_max must be positive";
    if (max_tick_age_ms < 0)        return "max_tick_age_ms must be non-negative";
    if (min_mid <= 0)               return "min_mid must be positive";
    if (competitive_max_slip_bps < 0.0) return "competitive_max_slip_bps must be non-negative";
    return {};
}

// ---------------------------------------------------------------------------
// calendar
// ---------------------------------------------------------------------------
namespace {
// Hours elapsed since Monday 00:00 UTC.
double week_hours(MarketClock c) noexcept {
    return static_cast<double>(c.weekday) * 24.0 + c.hour_utc;
}
constexpr double kFridayClose = 4.0 * 24.0 + kCloseHourUtc;  // Fri 22:00 -> 118
constexpr double kSundayOpen  = 6.0 * 24.0 + kOpenHourUtc;   // Sun 22:00 -> 166
}  // namespace

bool market_open(MarketClock c) noexcept {
    const double h = week_hours(c);
    return h < kFridayClose || h >= kSundayOpen;
}

double hours_to_open(MarketClock c) noexcept {
    if (market_open(c)) return 0.0;
    return kSundayOpen - week_hours(c);
}

double hours_to_close(MarketClock c) noexcept {
    if (!market_open(c)) return 0.0;
    const double h = week_hours(c);
    // After Sunday open the next close is the following Friday, one week on.
    if (h >= kSundayOpen) return (kFridayClose + 7.0 * 24.0) - h;
    return kFridayClose - h;
}

double hedge_horizon_hours(MarketClock c, double intraday_hours, double taper_hours) noexcept {
    if (intraday_hours < 0.0) intraday_hours = 0.0;
    if (!market_open(c)) {
        return intraday_hours + hours_to_open(c);
    }
    if (taper_hours <= 0.0) return intraday_hours;

    const double ttc = hours_to_close(c);
    double       w   = 1.0 - ttc / taper_hours;  // ramps 0 -> 1 into the close
    if (w < 0.0) w = 0.0;
    if (w > 1.0) w = 1.0;
    return intraday_hours + w * kWeekendGapHours;
}

}  // namespace vane
