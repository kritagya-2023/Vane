#include "vane/simulator.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace vane {

std::string SimConfig::validate() const {
    if (weeks <= 0) return "weeks must be positive";
    if (jitter_log_sd < 0.0) return "jitter_log_sd must be non-negative";
    if (intraday_horizon_hours < 0.0) return "intraday_horizon_hours must be non-negative";
    if (market.sigma_daily < 0.0) return "market.sigma_daily must be non-negative";
    if (market.mid0 <= 0) return "market.mid0 must be positive";
    if (market.comp_half_bps <= 0.0) return "market.comp_half_bps must be positive";
    if (const std::string e = pricing.validate(); !e.empty()) return "pricing: " + e;
    if (const std::string e = flow.validate(); !e.empty()) return "flow: " + e;
    if (const std::string e = desk.validate(); !e.empty()) return "desk: " + e;
    return {};
}

Simulator::Simulator(SimConfig cfg)
    : cfg_(std::move(cfg)),
      market_(cfg_.market),
      flow_(cfg_.flow),
      pricer_(cfg_.pricing),
      desk_(cfg_.desk),
      jitter_(cfg_.jitter_seed) {
    if (const std::string e = cfg_.validate(); !e.empty()) {
        throw std::invalid_argument("bad simulation config: " + e);
    }
}

void Simulator::run() {
    const SimTime horizon = cfg_.weeks * kSecondsPerWeek;
    events_.clear();
    oracle_.clear();

    std::int64_t next_id = 0;
    SimTime      t       = 0;

    while (true) {
        t = flow_.next_arrival(t);
        if (t >= horizon) break;

        market_.advance_to(t);

        const Points      mid = market_.mid();
        const MarketClock mc  = clock_at(t);

        // The client shops the rival first, and a buyer and a seller see
        // different halves of a sheet that may have gone stale in one
        // direction, so both anchors go in and the side draw picks one.
        const Customer c = flow_.make_customer(t, market_.comp_effective_half_bps(true),
                                               market_.comp_effective_half_bps(false));

        // --- build the quote ------------------------------------------------
        QuoteRequest req;
        req.mid                 = mid;
        req.inventory           = desk_.position();
        req.size                = c.size;
        req.tier                = c.tier;
        req.sigma_daily         = cfg_.market.sigma_daily;
        req.hedge_horizon_hours = hedge_horizon_hours(mc, cfg_.intraday_horizon_hours);
        req.tick_age_ms         = 0;
        req.competitor_bid      = market_.comp_bid();
        req.competitor_ask      = market_.comp_ask();

        // The policy chooses the spread; the engine still applies every clamp,
        // so a learned policy is held to exactly the same commercial limits as
        // the tier table it replaces.
        if (cfg_.policy) {
            PolicyContext pc;
            pc.req           = req;
            pc.comp_half_bps = c.comp_half_bps;
            pc.hour_utc      = mc.hour_utc;
            pc.market_open   = is_open(t);
            pc.client_buys   = c.client_buys;

            const Quote probe = pricer_.quote(req);
            if (probe.reject != RejectReason::None) continue;
            const double want = cfg_.policy->half_spread_bps(pc);
            // Expressed as a multiplier so the engine's clamp logic is the one
            // deciding what is admissible, not the policy.
            req.spread_multiplier =
                probe.half_spread_bps > 0.0 ? want / probe.half_spread_bps : 1.0;
        }

        const Quote unjittered = pricer_.quote(req);
        if (unjittered.reject != RejectReason::None) continue;

        const double policy_mult = req.spread_multiplier;
        const double mult = cfg_.jitter_log_sd > 0.0
                                ? std::exp(cfg_.jitter_log_sd * jitter_.normal())
                                : 1.0;
        req.spread_multiplier = policy_mult * mult;
        const Quote q         = pricer_.quote(req);

        // The relevant side must actually be showing.
        if (c.client_buys && !q.ask_valid) continue;
        if (!c.client_buys && !q.bid_valid) continue;

        const Points px = c.client_buys ? q.ask : q.bid;

        // --- the client decides --------------------------------------------
        const bool accepted = q.half_spread_bps <= c.reservation_half_bps;

        Event e;
        e.id            = next_id;
        e.t             = t;
        e.weekday       = mc.weekday;
        e.hour_utc      = mc.hour_utc;
        e.market_open   = is_open(t);
        e.tier          = c.tier;
        e.client_buys   = c.client_buys;
        e.size          = c.size;
        e.mid           = mid;
        e.inventory     = desk_.position();
        e.sigma_daily   = req.sigma_daily;
        e.horizon_h     = req.hedge_horizon_hours;
        e.comp_half_bps = c.comp_half_bps;
        e.base_half_bps = unjittered.half_spread_bps;
        e.spread_mult   = mult;
        e.jitter_log_sd = cfg_.jitter_log_sd;
        e.quoted_half_bps = q.half_spread_bps;
        e.clamped         = q.clamped_by_floor || q.clamped_by_ceiling || q.clamped_by_rival;
        e.quote_px        = px;
        e.accepted        = accepted;

        OracleEvent o;
        o.id                   = next_id;
        o.reservation_half_bps = c.reservation_half_bps;
        o.mu                   = c.mu;
        o.sigma_ln             = c.sigma_ln;
        o.accept_prob = accept_probability(c.mu, c.sigma_ln, q.half_spread_bps);
        o.oracle_half_bps =
            oracle_optimal_half_bps(c.mu, c.sigma_ln, cfg_.desk.hedge_cost_bps);
        o.oracle_margin_bps = expected_margin_bps(c.mu, c.sigma_ln, o.oracle_half_bps,
                                                  cfg_.desk.hedge_cost_bps);
        o.realised_margin_bps = expected_margin_bps(c.mu, c.sigma_ln, q.half_spread_bps,
                                                    cfg_.desk.hedge_cost_bps);

        events_.push_back(e);
        oracle_.push_back(o);
        ++next_id;

        if (accepted) {
            desk_.on_fill(c.client_buys, c.size, px, mid);
        }
        desk_.maybe_hedge(mid, t);
        desk_.mark(mid);
    }

    market_.advance_to(horizon);
    desk_.mark(market_.mid());
}

SimSummary Simulator::summary() const {
    SimSummary s;
    s.policy_name = cfg_.policy ? cfg_.policy->name() : "static";
    s.events = static_cast<std::int64_t>(events_.size());

    double sum_quoted = 0.0, sum_oracle = 0.0, sum_real_m = 0.0, sum_orac_m = 0.0;
    for (std::size_t i = 0; i < events_.size(); ++i) {
        if (events_[i].accepted) ++s.accepted;
        sum_quoted += events_[i].quoted_half_bps;
        sum_oracle += oracle_[i].oracle_half_bps;
        sum_real_m += oracle_[i].realised_margin_bps;
        sum_orac_m += oracle_[i].oracle_margin_bps;
    }
    if (s.events > 0) {
        const double n = static_cast<double>(s.events);
        s.hit_rate                 = static_cast<double>(s.accepted) / n;
        s.mean_quoted_half_bps     = sum_quoted / n;
        s.mean_oracle_half_bps     = sum_oracle / n;
        s.mean_realised_margin_bps = sum_real_m / n;
        s.mean_oracle_margin_bps   = sum_orac_m / n;
        s.regret_bps               = s.mean_oracle_margin_bps - s.mean_realised_margin_bps;
    }

    const DeskStats& d = desk_.stats();
    s.final_mid        = market_.mid();
    s.total_pnl        = desk_.equity(s.final_mid);
    s.gross_spread_pnl = d.gross_spread_pnl;
    s.hedge_cost       = d.hedge_cost;
    // Whatever the spread earned and hedging cost does not explain is the
    // market risk the desk carried on its inventory.
    s.inventory_pnl    = s.total_pnl - d.gross_spread_pnl + d.hedge_cost;
    s.client_volume    = d.client_volume;
    s.final_position   = desk_.position();
    s.max_abs_position = d.max_abs_position;
    s.max_drawdown     = d.max_drawdown;
    s.hedges           = d.hedges;
    return s;
}

void Simulator::write_events_csv(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::fprintf(f,
                 "id,t_sec,weekday,hour_utc,market_open,tier,client_buys,size,mid,"
                 "inventory,sigma_daily,horizon_h,comp_half_bps,base_half_bps,"
                 "spread_mult,jitter_log_sd,quoted_half_bps,clamped,quote_px,accepted\n");
    for (const Event& e : events_) {
        std::fprintf(f,
                     "%lld,%lld,%d,%.4f,%d,%s,%d,%lld,%.5f,%lld,%.6f,%.4f,%.4f,%.4f,"
                     "%.6f,%.4f,%.4f,%d,%.5f,%d\n",
                     static_cast<long long>(e.id), static_cast<long long>(e.t), e.weekday,
                     e.hour_utc, e.market_open ? 1 : 0,
                     std::string(tier_name(e.tier)).c_str(), e.client_buys ? 1 : 0,
                     static_cast<long long>(e.size), to_rate(e.mid),
                     static_cast<long long>(e.inventory), e.sigma_daily, e.horizon_h,
                     e.comp_half_bps, e.base_half_bps, e.spread_mult, e.jitter_log_sd,
                     e.quoted_half_bps, e.clamped ? 1 : 0, to_rate(e.quote_px),
                     e.accepted ? 1 : 0);
    }
    std::fclose(f);
}

void Simulator::write_oracle_csv(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::fprintf(f,
                 "id,reservation_half_bps,mu,sigma_ln,accept_prob,oracle_half_bps,"
                 "oracle_margin_bps,realised_margin_bps\n");
    for (const OracleEvent& o : oracle_) {
        std::fprintf(f, "%lld,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                     static_cast<long long>(o.id), o.reservation_half_bps, o.mu, o.sigma_ln,
                     o.accept_prob, o.oracle_half_bps, o.oracle_margin_bps,
                     o.realised_margin_bps);
    }
    std::fclose(f);
}

}  // namespace vane
