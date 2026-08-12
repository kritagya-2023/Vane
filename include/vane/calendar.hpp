// Vane - retail FX auto-quoter
// calendar.hpp : interbank trading week, and the risk horizon that follows
// from it.
//
// The interbank market runs 24/5 and shuts over the weekend, but retail demand
// does not. A position taken at 21:45 on Friday cannot be hedged until Sunday
// night, so its risk horizon is not the next few minutes but the whole weekend
// gap. That is the single input that makes the weekend widener fall out of the
// pricing formula rather than being bolted on as a special case.
#pragma once

namespace vane {

// Monday = 0 ... Sunday = 6, hour in UTC as a fractional value in [0, 24).
struct MarketClock {
    int    weekday  = 0;
    double hour_utc = 0.0;
};

inline constexpr double kWeekendGapHours = 48.0;  // Fri 22:00 -> Sun 22:00 UTC
inline constexpr double kOpenHourUtc     = 22.0;  // Sunday open
inline constexpr double kCloseHourUtc    = 22.0;  // Friday close

bool market_open(MarketClock c) noexcept;

// Hours until the market next opens. Zero while it is open.
double hours_to_open(MarketClock c) noexcept;

// Hours until the market next closes. Zero (or negative meaning) while closed;
// returns 0.0 when already closed.
double hours_to_close(MarketClock c) noexcept;

// The horizon over which the desk is exposed before it can next hedge.
//
//   open, far from close  -> intraday_hours
//   open, near the close  -> intraday_hours + w * 48h, w ramping 0 -> 1 over
//                            the final `taper_hours` of the week
//   closed                -> intraday_hours + hours until the market reopens
//
// Monotone non-increasing in time-to-close, which the property tests assert.
double hedge_horizon_hours(MarketClock c,
                           double      intraday_hours = 0.5,
                           double      taper_hours    = 8.0) noexcept;

}  // namespace vane
