#include "vane/flow.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

namespace vane {

FlowConfig FlowConfig::defaults() noexcept {
    FlowConfig c;
    // arrival_weight, size_log_med, size_log_sd, size_min, size_max, stickiness, sigma_ln
    //
    // size_log_med is ln(median ticket in USD): retail ~2,000, corporate
    // ~12,000, private ~35,000, wealth ~90,000.
    //
    // stickiness is the median reservation as a multiple of the competitor's
    // spread. Retail will pay well over the specialist rate for a bank they
    // already hold an account with; wealth clients barely will.
    c.tiers[0] = TierFlowParams{0.72, 7.60, 0.85,  100, 100'000, 1.75, 0.50};  // Retail
    c.tiers[1] = TierFlowParams{0.18, 9.39, 0.95,  500, 240'000, 1.30, 0.42};  // Corporate
    c.tiers[2] = TierFlowParams{0.07, 10.46, 0.80, 2'000, 240'000, 1.12, 0.36};  // Private
    c.tiers[3] = TierFlowParams{0.03, 11.41, 0.75, 5'000, 240'000, 1.04, 0.30};  // Wealth
    return c;
}

std::string FlowConfig::validate() const {
    std::ostringstream e;
    if (base_arrivals_per_day <= 0.0) return "base_arrivals_per_day must be positive";
    if (p_client_buys < 0.0 || p_client_buys > 1.0) return "p_client_buys must be in [0, 1]";
    if (size_ref <= 0) return "size_ref must be positive";
    if (hour_spread <= 0.0) return "hour_spread must be positive";
    if (hour_floor < 0.0 || hour_floor > 1.0) return "hour_floor must be in [0, 1]";
    double wsum = 0.0;
    for (int i = 0; i < kTierCount; ++i) {
        const TierFlowParams& t = tiers[i];
        std::string_view      n = tier_name(static_cast<Tier>(i));
        if (t.arrival_weight < 0.0) { e << "tier " << n << ": negative arrival_weight"; return e.str(); }
        if (t.size_log_sd <= 0.0)   { e << "tier " << n << ": size_log_sd must be positive"; return e.str(); }
        if (t.sigma_ln <= 0.0)      { e << "tier " << n << ": sigma_ln must be positive"; return e.str(); }
        if (t.stickiness <= 0.0)    { e << "tier " << n << ": stickiness must be positive"; return e.str(); }
        if (t.size_min <= 0 || t.size_max < t.size_min) {
            e << "tier " << n << ": bad size bounds";
            return e.str();
        }
        wsum += t.arrival_weight;
    }
    if (wsum <= 0.0) return "arrival weights sum to zero";
    for (int d = 0; d < 7; ++d)
        if (day_factor[d] < 0.0) return "day_factor must be non-negative";
    return {};
}

// ---------------------------------------------------------------------------
// demand curve
// ---------------------------------------------------------------------------

double accept_probability(double mu, double sigma_ln, double half_bps) noexcept {
    if (half_bps <= 0.0) return 1.0;
    if (sigma_ln <= 0.0) return std::log(half_bps) <= mu ? 1.0 : 0.0;
    return norm_cdf((mu - std::log(half_bps)) / sigma_ln);
}

double expected_margin_bps(double mu, double sigma_ln, double half_bps, double cost_bps) noexcept {
    const double net = half_bps - cost_bps;
    if (net <= 0.0) return 0.0;
    return net * accept_probability(mu, sigma_ln, half_bps);
}

double oracle_optimal_half_bps(double mu, double sigma_ln, double cost_bps) noexcept {
    // Search over ln(delta). The lower bound sits just above the cost, since
    // nothing below it can be profitable; the upper bound is far enough into
    // the tail that acceptance is negligible.
    const double lo = std::log(std::max(cost_bps, 1e-6) * 1.000001 + 1e-9);
    const double hi = mu + 8.0 * std::max(sigma_ln, 1e-6);
    if (hi <= lo) return std::exp(hi);

    auto f = [&](double log_d) { return expected_margin_bps(mu, sigma_ln, std::exp(log_d), cost_bps); };

    // Coarse grid first, so a non-unimodal objective cannot trap the refiner in
    // a local maximum.
    constexpr int kGrid = 512;
    double        best_x = lo, best_y = f(lo);
    for (int i = 1; i <= kGrid; ++i) {
        const double x = lo + (hi - lo) * static_cast<double>(i) / kGrid;
        const double y = f(x);
        if (y > best_y) { best_y = y; best_x = x; }
    }

    // Ternary refinement inside the winning cell.
    const double cell = (hi - lo) / kGrid;
    double       a    = std::max(lo, best_x - cell);
    double       b    = std::min(hi, best_x + cell);
    for (int i = 0; i < 200; ++i) {
        const double m1 = a + (b - a) / 3.0;
        const double m2 = b - (b - a) / 3.0;
        if (f(m1) < f(m2)) a = m1; else b = m2;
    }
    return std::exp(0.5 * (a + b));
}

// ---------------------------------------------------------------------------
// arrivals
// ---------------------------------------------------------------------------

FlowGenerator::FlowGenerator(FlowConfig cfg) : cfg_(cfg), rng_(cfg.seed) {
    double max_day = 0.0;
    for (int d = 0; d < 7; ++d) max_day = std::max(max_day, cfg_.day_factor[d]);
    // The hourly factor peaks at exactly 1.0 by construction.
    lambda_max_ = cfg_.base_arrivals_per_day * max_day / static_cast<double>(kSecondsPerDay);
}

double FlowGenerator::intensity(SimTime t) const noexcept {
    const MarketClock c = clock_at(t);
    double            h = c.hour_utc + 5.5;  // IST
    if (h >= 24.0) h -= 24.0;

    const double z    = (h - cfg_.peak_hour_ist) / cfg_.hour_spread;
    const double bump = std::exp(-0.5 * z * z);
    const double hour_factor = cfg_.hour_floor + (1.0 - cfg_.hour_floor) * bump;

    const double per_day = cfg_.base_arrivals_per_day * cfg_.day_factor[c.weekday] * hour_factor;
    return per_day / static_cast<double>(kSecondsPerDay);
}

SimTime FlowGenerator::next_arrival(SimTime from) noexcept {
    // Lewis-Shedler thinning: propose from a homogeneous process at the
    // dominating rate, then keep each proposal with probability lambda/lambda_max.
    double t = static_cast<double>(from);
    for (int guard = 0; guard < 1'000'000; ++guard) {
        t += rng_.exponential(lambda_max_);
        const SimTime ts = static_cast<SimTime>(t);
        if (rng_.uniform() * lambda_max_ <= intensity(ts)) {
            return ts > from ? ts : from + 1;
        }
    }
    return from + kSecondsPerDay;  // pathological config; caller will terminate
}

Customer FlowGenerator::make_customer(SimTime t, double comp_half_bps_buy,
                                      double comp_half_bps_sell) noexcept {
    double weights[kTierCount];
    for (int i = 0; i < kTierCount; ++i) weights[i] = cfg_.tiers[i].arrival_weight;

    Customer c;
    c.t    = t;
    c.tier = static_cast<Tier>(rng_.categorical(weights, kTierCount));
    const TierFlowParams& tp = cfg_.tiers[static_cast<int>(c.tier)];

    c.client_buys = rng_.bernoulli(cfg_.p_client_buys);

    const double raw = rng_.lognormal(tp.size_log_med, tp.size_log_sd);
    c.size = std::clamp(static_cast<Notional>(raw), tp.size_min, tp.size_max);

    c.comp_half_bps = c.client_buys ? comp_half_bps_buy : comp_half_bps_sell;
    c.sigma_ln      = tp.sigma_ln;
    c.mu = std::log(c.comp_half_bps) + std::log(tp.stickiness) +
           cfg_.beta_size * std::log(static_cast<double>(c.size) /
                                     static_cast<double>(cfg_.size_ref));
    c.reservation_half_bps = std::exp(c.mu + c.sigma_ln * rng_.normal());
    return c;
}

}  // namespace vane
