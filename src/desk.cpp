#include "vane/desk.hpp"

#include <cmath>
#include <cstdlib>

namespace vane {

std::string DeskConfig::validate() const {
    if (hedge_band < 0) return "hedge_band must be non-negative";
    if (hedge_cost_bps < 0.0) return "hedge_cost_bps must be non-negative";
    return {};
}

void Desk::on_fill(bool client_buys, Notional size, Points quote, Points mid) noexcept {
    if (size <= 0) return;
    const double px  = to_rate(quote);
    const double m   = to_rate(mid);
    const double qty = static_cast<double>(size);

    if (client_buys) {
        // Desk sells base currency at the ask: cash in, position down.
        cash_ += px * qty;
        position_ -= size;
    } else {
        // Desk buys base currency at the bid: cash out, position up.
        cash_ -= px * qty;
        position_ += size;
    }

    stats_.trades += 1;
    stats_.client_volume += size;
    stats_.gross_spread_pnl += std::fabs(px - m) * qty;
    const Notional abs_pos = std::llabs(position_);
    if (abs_pos > stats_.max_abs_position) stats_.max_abs_position = abs_pos;
}

Notional Desk::maybe_hedge(Points mid, SimTime t) noexcept {
    if (!cfg_.hedging_enabled) return 0;
    if (!is_open(t)) return 0;  // cannot hedge a closed market
    if (std::llabs(position_) <= cfg_.hedge_band) return 0;

    const Notional qty  = position_;  // unwind all the way to flat
    const double   m    = to_rate(mid);
    const double   cost = cfg_.hedge_cost_bps / kBpsScale;

    if (qty > 0) {
        // Long: sell into the market, paying half the interbank spread.
        const double px = m * (1.0 - cost);
        cash_ += px * static_cast<double>(qty);
        stats_.hedge_cost += m * cost * static_cast<double>(qty);
    } else {
        const double px = m * (1.0 + cost);
        cash_ += px * static_cast<double>(qty);  // qty negative: cash goes out
        stats_.hedge_cost += m * cost * static_cast<double>(-qty);
    }
    position_ = 0;
    stats_.hedges += 1;
    stats_.hedge_volume += std::llabs(qty);
    return qty;
}

void Desk::mark(Points mid) noexcept {
    const double eq = equity(mid);
    if (eq > stats_.peak_equity) stats_.peak_equity = eq;
    const double dd = stats_.peak_equity - eq;
    if (dd > stats_.max_drawdown) stats_.max_drawdown = dd;
}

}  // namespace vane
