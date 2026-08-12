// Vane - retail FX auto-quoter
// desk.hpp : position, cash and hedging.
//
// Accounting is deliberately primitive: one cash balance and one position, both
// updated on every fill. Total PnL is always cash + position * mid, so there is
// a single source of truth and no way for an attribution bucket to drift away
// from it. The attribution figures below are reported alongside, and a test
// asserts they reconcile.
#pragma once

#include <string>

#include "vane/market.hpp"
#include "vane/types.hpp"

namespace vane {

struct DeskConfig {
    // Hedge whenever the absolute position exceeds this, unwinding all the way
    // back to flat. Only possible while the interbank market is open, which is
    // precisely why weekend inventory has to be priced for.
    Notional hedge_band     = 400'000;
    double   hedge_cost_bps = 1.5;  // interbank spread plus brokerage
    bool     hedging_enabled = true;

    std::string validate() const;
};

struct DeskStats {
    std::int64_t trades          = 0;
    std::int64_t hedges          = 0;
    Notional     client_volume   = 0;   // total base ccy traded with clients
    Notional     hedge_volume    = 0;
    double       gross_spread_pnl = 0.0;  // sum of |quote - mid| * size
    double       hedge_cost       = 0.0;  // sum of hedge slippage paid
    Notional     max_abs_position = 0;
    double       peak_equity      = 0.0;
    double       max_drawdown     = 0.0;
};

class Desk {
public:
    explicit Desk(DeskConfig cfg = DeskConfig{}) noexcept : cfg_(cfg) {}

    // Applies a client fill. `client_buys` means the desk sells base currency
    // at `ask`; otherwise the desk buys at `bid`.
    void on_fill(bool client_buys, Notional size, Points quote, Points mid) noexcept;

    // Unwinds to flat if outside the band and the market is open. Returns the
    // notional hedged, signed the way the desk traded it.
    Notional maybe_hedge(Points mid, SimTime t) noexcept;

    // Marks the book and updates drawdown tracking.
    void mark(Points mid) noexcept;

    Notional position() const noexcept { return position_; }
    double   cash() const noexcept { return cash_; }
    double   equity(Points mid) const noexcept {
        return cash_ + static_cast<double>(position_) * to_rate(mid);
    }

    const DeskStats&  stats() const noexcept { return stats_; }
    const DeskConfig& config() const noexcept { return cfg_; }

private:
    DeskConfig cfg_;
    DeskStats  stats_;
    Notional   position_ = 0;
    double     cash_     = 0.0;
};

}  // namespace vane
