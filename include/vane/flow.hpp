// Vane - retail FX auto-quoter
// flow.hpp : where the customers come from and what they are willing to pay.
//
// The reservation spread is the whole point of this file. Each arriving client
// carries a private number: the widest half-spread, in bps, they would still
// deal at. The desk never sees it; it only ever sees whether the client dealt.
//
// That number is anchored on what the competitor is showing, multiplied by a
// tier stickiness factor. A retail traveller will pay a good deal more than the
// specialist's rate for a bank they already hold an account with; a wealth
// client will not. This reproduces the pricing gap in the internship snapshot
// as a behaviour rather than as an assumption, and it makes the demand curve
// analytic:
//
//     ln(reservation) ~ Normal(mu, s)
//     P(accept | delta) = Phi( (mu - ln delta) / s )
//
// which in turn means the revenue-maximising spread can be computed exactly.
// Phase 3 has to recover it from data alone; Phase 4 scores the gap as regret.
#pragma once

#include "vane/market.hpp"
#include "vane/random.hpp"
#include "vane/types.hpp"

namespace vane {

struct TierFlowParams {
    double   arrival_weight = 1.0;   // relative share of total arrivals
    double   size_log_med   = 7.6;   // ln(median ticket) in base-ccy units
    double   size_log_sd    = 0.8;
    Notional size_min       = 100;
    Notional size_max       = 240'000;
    double   stickiness     = 1.0;   // reservation / competitor spread, at the median
    double   sigma_ln       = 0.45;  // dispersion of willingness to pay
};

struct FlowConfig {
    // Arrivals per day across all tiers, before seasonality.
    double base_arrivals_per_day = 520.0;

    TierFlowParams tiers[kTierCount];

    // Indian retail demand is dominated by outbound flow: travel, tuition,
    // remittances. The desk therefore sells base currency far more often than
    // it buys, which builds a persistent short position and makes the inventory
    // skew earn its keep.
    double p_client_buys = 0.72;

    // Ticket size feeds price sensitivity: bigger tickets shop harder.
    double   beta_size = -0.11;
    Notional size_ref  = 5'000;

    // Seasonality. Hours are IST (UTC+5:30).
    double  peak_hour_ist  = 14.0;
    double  hour_spread    = 5.0;
    double  hour_floor     = 0.08;
    double  day_factor[7]  = {1.0, 1.0, 1.0, 1.0, 1.1, 0.75, 0.55};

    std::uint64_t seed = 2;

    static FlowConfig defaults() noexcept;
    std::string       validate() const;
};

struct Customer {
    SimTime  t            = 0;
    Tier     tier         = Tier::Retail;
    bool     client_buys  = true;   // true: client buys base ccy, desk sells at ask
    Notional size         = 0;

    // Hidden truth. Never written to the observable event log.
    double reservation_half_bps = 0.0;
    double mu                   = 0.0;  // ln-scale location of the reservation
    double sigma_ln             = 0.0;  // ln-scale dispersion
    double comp_half_bps        = 0.0;  // the anchor they were quoted against
};

// --- demand curve -----------------------------------------------------------

// P(client accepts a quoted half-spread of `half_bps`).
double accept_probability(double mu, double sigma_ln, double half_bps) noexcept;

// The half-spread maximising (delta - cost) * P(accept | delta). Solved by a
// log-spaced grid followed by ternary refinement, which does not assume
// unimodality the way a bare golden-section search would.
double oracle_optimal_half_bps(double mu, double sigma_ln, double cost_bps = 0.0) noexcept;

// Expected net revenue in bps per unit notional at a given quote.
double expected_margin_bps(double mu, double sigma_ln, double half_bps,
                           double cost_bps = 0.0) noexcept;

// --- arrival process --------------------------------------------------------

class FlowGenerator {
public:
    explicit FlowGenerator(FlowConfig cfg);

    // Total arrival intensity, in arrivals per second, at time t.
    double intensity(SimTime t) const noexcept;
    double max_intensity() const noexcept { return lambda_max_; }

    // Next arrival strictly after `from`, by Lewis-Shedler thinning.
    SimTime next_arrival(SimTime from) noexcept;

    // Draws everything about a client except their arrival time. Both anchors
    // are passed because the side is drawn inside, and a buyer and a seller see
    // different halves of the rival's (possibly stale) sheet.
    Customer make_customer(SimTime t, double comp_half_bps_buy,
                           double comp_half_bps_sell) noexcept;

    const FlowConfig& config() const noexcept { return cfg_; }

private:
    FlowConfig cfg_;
    Rng        rng_;
    double     lambda_max_ = 0.0;
};

}  // namespace vane
