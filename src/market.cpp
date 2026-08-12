#include "vane/market.hpp"

#include <algorithm>
#include <cmath>

namespace vane {

MarketClock clock_at(SimTime t) noexcept {
    const SimTime rem = ((t % kSecondsPerWeek) + kSecondsPerWeek) % kSecondsPerWeek;
    MarketClock   c;
    c.weekday  = static_cast<int>(rem / kSecondsPerDay);
    c.hour_utc = static_cast<double>(rem % kSecondsPerDay) / 3600.0;
    return c;
}

bool is_open(SimTime t) noexcept {
    const SimTime rem = ((t % kSecondsPerWeek) + kSecondsPerWeek) % kSecondsPerWeek;
    return rem < kFridayCloseSec || rem >= kSundayOpenSec;
}

SimTime cumulative_open_seconds(SimTime t) noexcept {
    if (t <= 0) return 0;
    const SimTime weeks = t / kSecondsPerWeek;
    const SimTime rem   = t % kSecondsPerWeek;
    SimTime       total = weeks * (kSecondsPerWeek - kWeekendSec);
    if (rem <= kFridayCloseSec) {
        total += rem;
    } else if (rem <= kSundayOpenSec) {
        total += kFridayCloseSec;
    } else {
        total += kFridayCloseSec + (rem - kSundayOpenSec);
    }
    return total;
}

SimTime open_seconds_between(SimTime a, SimTime b) noexcept {
    if (b <= a) return 0;
    return cumulative_open_seconds(b) - cumulative_open_seconds(a);
}

namespace {
// floor division that behaves for negative numerators.
std::int64_t floor_div(std::int64_t n, std::int64_t d) noexcept {
    std::int64_t q = n / d;
    if ((n % d != 0) && ((n < 0) != (d < 0))) --q;
    return q;
}
}  // namespace

int reopens_between(SimTime a, SimTime b) noexcept {
    if (b <= a) return 0;
    const std::int64_t ka = floor_div(a - kSundayOpenSec, kSecondsPerWeek);
    const std::int64_t kb = floor_div(b - kSundayOpenSec, kSecondsPerWeek);
    return static_cast<int>(kb - ka);
}

// ---------------------------------------------------------------------------

std::string MarketConfig::validate() const {
    if (mid0 <= 0) return "mid0 must be positive";
    if (sigma_daily < 0.0) return "sigma_daily must be non-negative";
    if (weekend_vol_mult < 0.0) return "weekend_vol_mult must be non-negative";
    if (path_step_sec <= 0) return "path_step_sec must be positive";
    if (comp_half_bps <= 0.0) return "comp_half_bps must be positive";
    if (comp_min_eff_half_bps < 0.0) return "comp_min_eff_half_bps must be non-negative";
    for (const double h : comp_refresh_hours)
        if (h < 0.0 || h >= 24.0) return "comp_refresh_hours must be in [0, 24)";
    return {};
}

Market::Market(MarketConfig cfg) : cfg_(std::move(cfg)), rng_(cfg_.seed) {
    const double h   = cfg_.comp_half_bps / kBpsScale;
    const double mid = to_rate(cfg_.mid0);
    path_.push_back(cfg_.mid0);
    sheet_bid_.push_back(floor_to_points(mid * (1.0 - h)));
    sheet_ask_.push_back(ceil_to_points(mid * (1.0 + h)));
}

std::size_t Market::index_of(SimTime t) const noexcept {
    if (t <= 0) return 0;
    return static_cast<std::size_t>(t / cfg_.path_step_sec);
}

// True if a rate-sheet refresh instant falls inside grid step k, that is in
// ((k-1)*step, k*step].
bool Market::refresh_in_step(std::size_t k) const noexcept {
    const SimTime lo = static_cast<SimTime>(k - 1) * cfg_.path_step_sec;
    const SimTime hi = static_cast<SimTime>(k) * cfg_.path_step_sec;
    for (SimTime day_start = (lo / kSecondsPerDay) * kSecondsPerDay;
         day_start <= hi; day_start += kSecondsPerDay) {
        for (const double hr : cfg_.comp_refresh_hours) {
            const SimTime inst = day_start + static_cast<SimTime>(hr * 3600.0);
            if (inst > lo && inst <= hi && is_open(inst)) return true;
        }
    }
    return false;
}

void Market::ensure_index(std::size_t k) const {
    while (path_.size() <= k) {
        const std::size_t j  = path_.size();          // index being generated
        const SimTime     t0 = static_cast<SimTime>(j - 1) * cfg_.path_step_sec;
        const SimTime     t1 = static_cast<SimTime>(j) * cfg_.path_step_sec;

        const SimTime open_sec = open_seconds_between(t0, t1);
        const int     reopens  = reopens_between(t0, t1);

        double log_ret = 0.0;
        if (open_sec > 0 && cfg_.sigma_daily > 0.0) {
            const double dt = static_cast<double>(open_sec) / static_cast<double>(kSecondsPerDay);
            const double s  = cfg_.sigma_daily;
            log_ret += -0.5 * s * s * dt + s * std::sqrt(dt) * rng_.normal();
        }
        // Each weekend is a single jump carrying 48 hours of variance, scaled
        // for gap risk. The closed period contributes no diffusion, so nothing
        // is double counted.
        for (int i = 0; i < reopens && cfg_.sigma_daily > 0.0; ++i) {
            const double sw = cfg_.sigma_daily * cfg_.weekend_vol_mult;
            const double dt =
                static_cast<double>(kWeekendSec) / static_cast<double>(kSecondsPerDay);
            log_ret += -0.5 * sw * sw * dt + sw * std::sqrt(dt) * rng_.normal();
        }

        const Points prev = path_.back();
        const Points next =
            log_ret == 0.0
                ? prev
                : std::max<Points>(1, round_to_points(to_rate(prev) * std::exp(log_ret)));
        path_.push_back(next);

        if (refresh_in_step(j)) {
            const double h = cfg_.comp_half_bps / kBpsScale;
            const double m = to_rate(next);
            sheet_bid_.push_back(floor_to_points(m * (1.0 - h)));
            sheet_ask_.push_back(ceil_to_points(m * (1.0 + h)));
        } else {
            sheet_bid_.push_back(sheet_bid_.back());
            sheet_ask_.push_back(sheet_ask_.back());
        }
    }
}

Points Market::mid_at(SimTime t) const {
    const std::size_t k = index_of(t);
    ensure_index(k);
    return path_[k];
}

Points Market::comp_bid() const {
    const std::size_t k = index_of(now_);
    ensure_index(k);
    return sheet_bid_[k];
}

Points Market::comp_ask() const {
    const std::size_t k = index_of(now_);
    ensure_index(k);
    return sheet_ask_[k];
}

void Market::advance_to(SimTime t) {
    if (t > now_) now_ = t;
    ensure_index(index_of(now_));
}

double Market::comp_effective_half_bps(bool client_buys) const {
    const double m0 = to_rate(mid());
    const double eff = client_buys ? (to_rate(comp_ask()) - m0) : (m0 - to_rate(comp_bid()));
    const double bps = eff / m0 * kBpsScale;
    return std::max(bps, cfg_.comp_min_eff_half_bps);
}

}  // namespace vane
