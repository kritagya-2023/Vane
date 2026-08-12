// Vane - retail FX auto-quoter
// market.hpp : the world outside the desk.
//
// The mid-price process diffuses only while the interbank market is open and
// jumps across the weekend. The competitor's rate sheet is refreshed a couple
// of times a day and then goes stale, exactly as a published bank sheet does.
//
// The path is generated on a fixed time grid, lazily extended, and never as a
// function of when it is queried. That matters: if the price were drawn at
// arrival times, changing the customer-flow seed would silently change the
// market too, and no two policies could ever be compared on the same world.
// With a fixed grid, Phase 4 can run competing quoters against an identical
// price path and attribute the whole difference to the policy.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "vane/calendar.hpp"
#include "vane/random.hpp"
#include "vane/types.hpp"

namespace vane {

// Seconds since Monday 00:00:00 UTC of the first simulated week.
using SimTime = std::int64_t;

inline constexpr SimTime kSecondsPerDay  = 86'400;
inline constexpr SimTime kSecondsPerWeek = 7 * kSecondsPerDay;
inline constexpr SimTime kFridayCloseSec = 4 * kSecondsPerDay + 22 * 3600;  // 424800
inline constexpr SimTime kSundayOpenSec  = 6 * kSecondsPerDay + 22 * 3600;  // 597600
inline constexpr SimTime kWeekendSec     = kSundayOpenSec - kFridayCloseSec;

MarketClock clock_at(SimTime t) noexcept;
bool        is_open(SimTime t) noexcept;

// Cumulative seconds of open market between time 0 and t.
SimTime cumulative_open_seconds(SimTime t) noexcept;
SimTime open_seconds_between(SimTime a, SimTime b) noexcept;

// Number of Sunday reopens strictly after `a` and at or before `b`.
int reopens_between(SimTime a, SimTime b) noexcept;

struct MarketConfig {
    Points mid0             = 9'529'000;  // USD/INR 95.29
    double sigma_daily      = 0.0040;     // 40 bps per open day
    double weekend_vol_mult = 1.30;       // gap risk exceeds ordinary diffusion

    // Resolution of the price grid. Quotes are struck off the most recent grid
    // point, which is also how a desk works off the last tick it received.
    SimTime path_step_sec = 60;

    // Competitor: a specialist publishing a static sheet, refreshed at these
    // UTC hours on open days. 04:00 and 09:30 UTC are 09:30 and 15:00 IST.
    double              comp_half_bps         = 130.0;
    std::vector<double> comp_refresh_hours    = {4.0, 9.5};
    double              comp_min_eff_half_bps = 5.0;

    std::uint64_t seed = 1;

    std::string validate() const;
};

class Market {
public:
    explicit Market(MarketConfig cfg);

    // Moves the world forward. Times in the past are ignored.
    void advance_to(SimTime t);

    SimTime now() const noexcept { return now_; }
    Points  mid() const noexcept { return mid_at(now_); }
    Points  comp_bid() const;
    Points  comp_ask() const;

    // Price at an arbitrary time. Const because the path is extended through a
    // mutable cache: the values returned never depend on the access order.
    Points mid_at(SimTime t) const;

    // What the customer sees on the rival's board, relative to the live mid. A
    // stale sheet can drift towards zero or blow out; floored so the demand
    // model always has a positive anchor.
    double comp_effective_half_bps(bool client_buys) const;

    const MarketConfig& config() const noexcept { return cfg_; }

private:
    std::size_t index_of(SimTime t) const noexcept;
    void        ensure_index(std::size_t k) const;
    bool        refresh_in_step(std::size_t k) const noexcept;

    MarketConfig cfg_;
    SimTime      now_ = 0;

    // Lazily grown, but always generated in grid order from the seed, so the
    // contents are a pure function of the configuration alone.
    mutable Rng                 rng_;
    mutable std::vector<Points> path_;
    mutable std::vector<Points> sheet_bid_;
    mutable std::vector<Points> sheet_ask_;
};

}  // namespace vane
