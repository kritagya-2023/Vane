// Randomised property tests.
//
// These assert the structural claims the pricing model makes, over tens of
// thousands of random inputs, rather than the handful of worked examples in
// test_unit.cpp. If a future change to the formula breaks monotonicity in
// volatility or the ordering of the tier table, this is what catches it.
#include <cstdint>

#include "harness.hpp"
#include "vane/pricer.hpp"

using namespace vane;

namespace {

// xorshift64*, so the corpus is reproducible without pulling in <random>.
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    std::uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
    double uniform(double lo, double hi) {
        const double u = static_cast<double>(next() >> 11) / 9007199254740992.0;
        return lo + u * (hi - lo);
    }
    std::int64_t integer(std::int64_t lo, std::int64_t hi) {
        return lo + static_cast<std::int64_t>(next() % static_cast<std::uint64_t>(hi - lo + 1));
    }
};

QuoteRequest random_request(Rng& g) {
    QuoteRequest r;
    r.mid                 = g.integer(5'000'000, 15'000'000);
    r.inventory           = g.integer(-3'000'000, 3'000'000);
    r.size                = g.integer(100, 200'000);
    r.tier                = static_cast<Tier>(g.integer(0, kTierCount - 1));
    r.sigma_daily         = g.uniform(0.0005, 0.02);
    r.hedge_horizon_hours = g.uniform(0.0, 50.0);
    r.tick_age_ms         = 0;
    return r;
}

constexpr int kIters = 20'000;

void prop_two_sided_and_ordered() {
    vt::section("property / bid < reservation < ask");
    Pricer p;
    Rng    g(0xC0FFEE);
    int    bad = 0;
    for (int i = 0; i < kIters; ++i) {
        const QuoteRequest r = random_request(g);
        const Quote        q = p.quote(r);
        if (q.reject != RejectReason::None) continue;
        if (!(q.bid < q.reservation && q.reservation < q.ask && q.bid < q.ask && q.spread() > 0))
            ++bad;
    }
    CHECK_MSG(bad == 0, "quote ordering violated in " + std::to_string(bad) + " cases");
}

void prop_spread_monotone_in_vol() {
    vt::section("property / spread non-decreasing in volatility");
    Pricer p;
    Rng    g(0xBEEF01);
    int    bad = 0;
    for (int i = 0; i < kIters; ++i) {
        QuoteRequest a = random_request(g);
        a.sigma_daily  = g.uniform(0.0005, 0.008);
        QuoteRequest b = a;
        b.sigma_daily  = a.sigma_daily + g.uniform(0.0001, 0.010);
        const Quote qa = p.quote(a);
        const Quote qb = p.quote(b);
        if (qa.reject != RejectReason::None || qb.reject != RejectReason::None) continue;
        if (qb.half_spread_bps < qa.half_spread_bps - 1e-9) ++bad;
    }
    CHECK_MSG(bad == 0, "vol monotonicity violated in " + std::to_string(bad) + " cases");
}

void prop_spread_monotone_in_horizon() {
    vt::section("property / spread non-decreasing in hedge horizon");
    Pricer p;
    Rng    g(0xBEEF02);
    int    bad = 0;
    for (int i = 0; i < kIters; ++i) {
        QuoteRequest a        = random_request(g);
        a.hedge_horizon_hours = g.uniform(0.0, 12.0);
        QuoteRequest b        = a;
        b.hedge_horizon_hours = a.hedge_horizon_hours + g.uniform(0.01, 40.0);
        const Quote qa        = p.quote(a);
        const Quote qb        = p.quote(b);
        if (qa.reject != RejectReason::None || qb.reject != RejectReason::None) continue;
        if (qb.sigma_horizon_bps < qa.sigma_horizon_bps - 1e-9) ++bad;
        if (qb.half_spread_bps < qa.half_spread_bps - 1e-9) ++bad;
    }
    CHECK_MSG(bad == 0, "horizon monotonicity violated in " + std::to_string(bad) + " cases");
}

void prop_skew_monotone_in_inventory() {
    vt::section("property / both sides shade down as inventory grows");
    Pricer p;
    Rng    g(0xBEEF03);
    int    bad = 0;
    for (int i = 0; i < kIters; ++i) {
        QuoteRequest a = random_request(g);
        a.inventory    = g.integer(-1'000'000, 1'000'000);
        a.size         = g.integer(100, 50'000);
        QuoteRequest b = a;
        b.inventory    = a.inventory + g.integer(1, 1'000'000);
        const Quote qa = p.quote(a);
        const Quote qb = p.quote(b);
        if (!qa.two_sided() || !qb.two_sided()) continue;
        // Longer inventory => weakly lower reservation, bid and ask.
        if (qb.skew_bps > qa.skew_bps + 1e-9) ++bad;
        if (qb.reservation > qa.reservation) ++bad;
        if (qb.bid > qa.bid || qb.ask > qa.ask) ++bad;
    }
    CHECK_MSG(bad == 0, "inventory skew monotonicity violated in " + std::to_string(bad) + " cases");
}

void prop_skew_does_not_widen() {
    vt::section("property / skew translates the price, it does not widen it");
    Pricer p;
    Rng    g(0xBEEF04);
    int    bad = 0;
    for (int i = 0; i < kIters; ++i) {
        QuoteRequest a = random_request(g);
        a.inventory    = 0;
        QuoteRequest b = a;
        b.inventory    = g.integer(-2'000'000, 2'000'000);
        const Quote qa = p.quote(a);
        const Quote qb = p.quote(b);
        if (!qa.two_sided() || !qb.two_sided()) continue;
        if (std::fabs(qa.half_spread_bps - qb.half_spread_bps) > 1e-9) ++bad;
    }
    CHECK_MSG(bad == 0, "skew altered the half-spread in " + std::to_string(bad) + " cases");
}

void prop_tier_ordering() {
    vt::section("property / tier ordering preserved for identical flow");
    Pricer p;
    Rng    g(0xBEEF05);
    int    bad = 0;
    for (int i = 0; i < kIters / 4; ++i) {
        QuoteRequest r = random_request(g);
        double       prev = 1e18;
        for (int t = 0; t < kTierCount; ++t) {
            r.tier        = static_cast<Tier>(t);
            const Quote q = p.quote(r);
            if (q.reject != RejectReason::None) break;
            if (q.half_spread_bps > prev + 1e-9) ++bad;
            prev = q.half_spread_bps;
        }
    }
    CHECK_MSG(bad == 0, "tier ordering violated in " + std::to_string(bad) + " cases");
}

void prop_floor_respected() {
    vt::section("property / economic floor and tier band always respected");
    Pricer p;
    Rng    g(0xBEEF06);
    int    bad = 0;
    for (int i = 0; i < kIters; ++i) {
        const QuoteRequest r = random_request(g);
        const Quote        q = p.quote(r);
        if (q.reject != RejectReason::None) continue;
        const TierParams& tp = p.config().tiers[static_cast<int>(r.tier)];
        if (q.half_spread_bps < p.config().cost_floor_bps - 1e-9) ++bad;
        if (q.half_spread_bps < tp.floor_half_bps - 1e-9) ++bad;
        if (q.half_spread_bps > tp.ceil_half_bps + 1e-9) ++bad;
        if (std::fabs(q.skew_bps) > p.config().max_skew_bps + 1e-9) ++bad;
    }
    CHECK_MSG(bad == 0, "spread bounds violated in " + std::to_string(bad) + " cases");
}

void prop_rounding_direction() {
    vt::section("property / rounding never favours the client");
    Pricer p;
    Rng    g(0xBEEF07);
    int    bad = 0;
    for (int i = 0; i < kIters; ++i) {
        const QuoteRequest r = random_request(g);
        const Quote        q = p.quote(r);
        if (q.reject != RejectReason::None) continue;
        const double res  = to_rate(q.reservation);
        const double half = q.half_spread_bps / kBpsScale;
        // Allow one point of slack for the reservation price's own rounding.
        if (to_rate(q.bid) > res * (1.0 - half) + 2.0 / kPointScale) ++bad;
        if (to_rate(q.ask) < res * (1.0 + half) - 2.0 / kPointScale) ++bad;
    }
    CHECK_MSG(bad == 0, "rounding direction violated in " + std::to_string(bad) + " cases");
}

void prop_no_crash_on_extremes() {
    vt::section("property / extreme inputs are rejected, never undefined");
    Pricer p;
    Rng    g(0xBEEF08);
    int    bad = 0;
    for (int i = 0; i < kIters; ++i) {
        QuoteRequest r = random_request(g);
        switch (g.integer(0, 5)) {
            case 0: r.mid = g.integer(-1'000, 20'000); break;
            case 1: r.sigma_daily = g.uniform(-1.0, 5.0); break;
            case 2: r.hedge_horizon_hours = g.uniform(-10.0, 5'000.0); break;
            case 3: r.size = g.integer(-1'000, 10'000'000); break;
            case 4: r.inventory = g.integer(-1'000'000'000, 1'000'000'000); break;
            default: r.tick_age_ms = g.integer(0, 100'000); break;
        }
        const Quote q = p.quote(r);
        if (q.reject != RejectReason::None) continue;
        if (!std::isfinite(q.half_spread_bps) || !std::isfinite(q.skew_bps)) ++bad;
        if (q.bid <= 0 || q.ask <= 0 || q.bid >= q.ask) ++bad;
    }
    CHECK_MSG(bad == 0, "extreme input produced a bad quote in " + std::to_string(bad) + " cases");
}

}  // namespace

int main() {
    prop_two_sided_and_ordered();
    prop_spread_monotone_in_vol();
    prop_spread_monotone_in_horizon();
    prop_skew_monotone_in_inventory();
    prop_skew_does_not_widen();
    prop_tier_ordering();
    prop_floor_respected();
    prop_rounding_direction();
    prop_no_crash_on_extremes();
    return vt::report("property");
}
