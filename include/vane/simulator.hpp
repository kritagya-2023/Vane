// Vane - retail FX auto-quoter
// simulator.hpp : the loop that ties market, flow, pricer and desk together.
//
// Its output is a labelled dataset. Every quote the desk shows is recorded
// along with whether the client dealt, which is exactly the training signal
// Phase 3 needs and exactly what no public dataset contains.
//
// The event log is split in two on purpose:
//
//   Event       - only what a real desk could observe at quote time.
//   OracleEvent - the client's hidden reservation spread and the true optimal
//                 quote, for scoring afterwards.
//
// They are written to separate files so that Phase 3 cannot accidentally train
// on the answer.
#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "vane/desk.hpp"
#include "vane/flow.hpp"
#include "vane/market.hpp"
#include "vane/policy.hpp"
#include "vane/pricer.hpp"

namespace vane {

// What the desk sees.
struct Event {
    std::int64_t id           = 0;
    SimTime      t            = 0;
    int          weekday      = 0;
    double       hour_utc     = 0.0;
    bool         market_open  = true;
    Tier         tier         = Tier::Retail;
    bool         client_buys  = true;
    Notional     size         = 0;
    Points       mid          = 0;
    Notional     inventory    = 0;   // before the trade
    double       sigma_daily  = 0.0;
    double       horizon_h    = 0.0;
    double       comp_half_bps = 0.0;

    double base_half_bps   = 0.0;  // what the policy wanted to quote
    double spread_mult     = 1.0;  // exploration jitter that was applied
    double jitter_log_sd   = 0.0;  // its scale, for propensity weighting
    double quoted_half_bps = 0.0;  // what was actually shown, post-clamp
    bool   clamped         = false;

    Points quote_px = 0;
    bool   accepted = false;
};

// What only the simulator knows.
struct OracleEvent {
    std::int64_t id                  = 0;
    double       reservation_half_bps = 0.0;
    double       mu                  = 0.0;
    double       sigma_ln            = 0.0;
    double       accept_prob         = 0.0;  // at the quote actually shown
    double       oracle_half_bps     = 0.0;  // revenue-maximising quote
    double       oracle_margin_bps   = 0.0;  // expected margin there
    double       realised_margin_bps = 0.0;  // expected margin at the quote shown
};

struct SimConfig {
    MarketConfig  market;
    FlowConfig    flow    = FlowConfig::defaults();
    PricingConfig pricing = PricingConfig::defaults();
    DeskConfig    desk;

    SimTime weeks = 4;

    // Exploration. Log-normal jitter with zero mean in log space, so the median
    // quote is unchanged and the policy stays commercially sane while still
    // generating the spread variation Phase 3 needs to identify demand.
    double        jitter_log_sd = 0.18;
    std::uint64_t jitter_seed   = 3;

    // Intraday hedging horizon handed to the pricer.
    double intraday_horizon_hours = 0.5;

    // Optional learned policy. When null the Phase 1 tier table is used, which
    // is the incumbent baseline every comparison is made against.
    std::shared_ptr<QuotePolicy> policy;

    std::string validate() const;
};

struct SimSummary {
    std::int64_t events        = 0;
    std::int64_t accepted      = 0;
    double       hit_rate      = 0.0;
    double       total_pnl     = 0.0;
    double       gross_spread_pnl = 0.0;
    double       hedge_cost    = 0.0;
    double       inventory_pnl = 0.0;  // total minus spread, plus hedge cost
    Notional     client_volume = 0;
    Notional     final_position = 0;
    Notional     max_abs_position = 0;
    double       max_drawdown  = 0.0;
    std::int64_t hedges        = 0;
    double       mean_quoted_half_bps = 0.0;
    double       mean_oracle_half_bps = 0.0;
    double       mean_realised_margin_bps = 0.0;
    double       mean_oracle_margin_bps   = 0.0;
    double       regret_bps    = 0.0;  // oracle margin minus realised margin
    Points       final_mid     = 0;
    std::string  policy_name   = "static";
};

class Simulator {
public:
    explicit Simulator(SimConfig cfg);

    void run();

    const std::vector<Event>&       events() const noexcept { return events_; }
    const std::vector<OracleEvent>& oracle() const noexcept { return oracle_; }
    SimSummary                      summary() const;

    void write_events_csv(const std::string& path) const;
    void write_oracle_csv(const std::string& path) const;

private:
    SimConfig                cfg_;
    Market                   market_;
    FlowGenerator            flow_;
    Pricer                   pricer_;
    Desk                     desk_;
    Rng                      jitter_;
    std::vector<Event>       events_;
    std::vector<OracleEvent> oracle_;
};

}  // namespace vane
