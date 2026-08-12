// Phase 2 tests.
//
// A simulator cannot be verified by worked examples alone: most of its claims
// are distributional. So alongside the ordinary unit checks there are
// statistical checks that draw large samples and compare against the analytic
// truth, and conservation checks that assert the accounting cannot drift.
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "harness.hpp"
#include "vane/desk.hpp"
#include "vane/flow.hpp"
#include "vane/market.hpp"
#include "vane/random.hpp"
#include "vane/simulator.hpp"

using namespace vane;

namespace {

// ---------------------------------------------------------------------------
void test_rng() {
    vt::section("rng / distributions");

    Rng g(12345);

    // Same seed, same stream.
    Rng a(999), b(999);
    bool same = true;
    for (int i = 0; i < 1000; ++i)
        if (a.next_u64() != b.next_u64()) same = false;
    CHECK(same);

    Rng c(1000);
    CHECK(Rng(999).next_u64() != c.next_u64());

    // uniform() stays strictly inside (0, 1) so logs never blow up.
    double lo = 1.0, hi = 0.0, sum = 0.0;
    for (int i = 0; i < 200'000; ++i) {
        const double u = g.uniform();
        lo = std::min(lo, u);
        hi = std::max(hi, u);
        sum += u;
    }
    CHECK(lo > 0.0);
    CHECK(hi < 1.0);
    CHECK(vt::near(sum / 200'000.0, 0.5, 0.005));

    // normal(): mean 0, variance 1, and roughly the right tail mass.
    double m = 0.0, m2 = 0.0;
    long   tail = 0;
    constexpr int kN = 400'000;
    for (int i = 0; i < kN; ++i) {
        const double z = g.normal();
        m += z;
        m2 += z * z;
        if (std::fabs(z) > 1.959963985) ++tail;
    }
    m /= kN;
    m2 /= kN;
    CHECK(vt::near(m, 0.0, 0.01));
    CHECK(vt::near(m2, 1.0, 0.01));
    CHECK(vt::near(static_cast<double>(tail) / kN, 0.05, 0.003));

    // lognormal(): median is exp(mu).
    std::vector<double> xs;
    xs.reserve(100'000);
    for (int i = 0; i < 100'000; ++i) xs.push_back(g.lognormal(2.0, 0.5));
    std::sort(xs.begin(), xs.end());
    CHECK(vt::near(xs[xs.size() / 2], std::exp(2.0), 0.1));

    // exponential(rate): mean 1/rate.
    double es = 0.0;
    for (int i = 0; i < 200'000; ++i) es += g.exponential(4.0);
    CHECK(vt::near(es / 200'000.0, 0.25, 0.005));

    // categorical() respects the weights.
    const double w[3] = {1.0, 3.0, 6.0};
    long          cnt[3]{};
    for (int i = 0; i < 200'000; ++i) ++cnt[g.categorical(w, 3)];
    CHECK(vt::near(static_cast<double>(cnt[0]) / 200'000.0, 0.1, 0.005));
    CHECK(vt::near(static_cast<double>(cnt[1]) / 200'000.0, 0.3, 0.006));
    CHECK(vt::near(static_cast<double>(cnt[2]) / 200'000.0, 0.6, 0.007));

    // norm_cdf against known values.
    CHECK(vt::near(norm_cdf(0.0), 0.5, 1e-12));
    CHECK(vt::near(norm_cdf(1.0), 0.8413447460685429, 1e-12));
    CHECK(vt::near(norm_cdf(-1.959963985), 0.025, 1e-9));
    CHECK(norm_cdf(-40.0) >= 0.0);
    CHECK(norm_cdf(40.0) <= 1.0);
}

// ---------------------------------------------------------------------------
void test_sim_calendar() {
    vt::section("market / simulation clock");

    CHECK(clock_at(0).weekday == 0 && vt::near(clock_at(0).hour_utc, 0.0));
    CHECK(clock_at(3600 * 9).weekday == 0 && vt::near(clock_at(3600 * 9).hour_utc, 9.0));
    CHECK(clock_at(kSecondsPerWeek + 3600).weekday == 0);
    CHECK(clock_at(4 * kSecondsPerDay + 3600 * 21).weekday == 4);

    CHECK(is_open(0));
    CHECK(is_open(kFridayCloseSec - 1));
    CHECK(!is_open(kFridayCloseSec));
    CHECK(!is_open(kSundayOpenSec - 1));
    CHECK(is_open(kSundayOpenSec));
    CHECK(is_open(kSecondsPerWeek + 100));
    CHECK(!is_open(kSecondsPerWeek + kFridayCloseSec + 100));

    // A full week contains exactly five open days.
    CHECK(cumulative_open_seconds(kSecondsPerWeek) == kSecondsPerWeek - kWeekendSec);
    CHECK(cumulative_open_seconds(3 * kSecondsPerWeek) == 3 * (kSecondsPerWeek - kWeekendSec));
    CHECK(cumulative_open_seconds(0) == 0);
    CHECK(cumulative_open_seconds(kFridayCloseSec) == kFridayCloseSec);
    // Nothing accrues while shut.
    CHECK(cumulative_open_seconds(kSundayOpenSec) == kFridayCloseSec);
    CHECK(cumulative_open_seconds(kFridayCloseSec + 3600) == kFridayCloseSec);

    CHECK(open_seconds_between(0, kSecondsPerWeek) == kSecondsPerWeek - kWeekendSec);
    CHECK(open_seconds_between(kFridayCloseSec, kSundayOpenSec) == 0);
    CHECK(open_seconds_between(100, 50) == 0);

    // Monotone and never exceeding wall-clock time.
    SimTime prev = 0;
    for (SimTime t = 0; t < 3 * kSecondsPerWeek; t += 997) {
        const SimTime o = cumulative_open_seconds(t);
        CHECK_MSG(o >= prev, "cumulative open seconds must be monotone");
        CHECK_MSG(o <= t, "open seconds cannot exceed elapsed seconds");
        prev = o;
    }

    CHECK(reopens_between(0, kSecondsPerWeek) == 1);
    CHECK(reopens_between(0, kSundayOpenSec - 1) == 0);
    CHECK(reopens_between(0, kSundayOpenSec) == 1);
    CHECK(reopens_between(0, 4 * kSecondsPerWeek) == 4);
    CHECK(reopens_between(kSundayOpenSec, kSundayOpenSec + kSecondsPerWeek) == 1);
    CHECK(reopens_between(500, 400) == 0);
}

// ---------------------------------------------------------------------------
void test_market_process() {
    vt::section("market / price process and rate sheet");

    // Zero volatility: the mid must not move at all.
    MarketConfig mcv;
    CHECK_MSG(mcv.validate().empty(), "market defaults should validate: " + mcv.validate());
    MarketConfig mbad = mcv;
    mbad.path_step_sec = 0;
    CHECK(!mbad.validate().empty());
    mbad = mcv;
    mbad.comp_half_bps = 0.0;
    CHECK(!mbad.validate().empty());

    MarketConfig mc;
    mc.sigma_daily = 0.0;
    Market flat(mc);
    flat.advance_to(3 * kSecondsPerWeek);
    CHECK(flat.mid() == mc.mid0);

    // advance_to must ignore times in the past.
    Market m2(mc);
    m2.advance_to(1000);
    m2.advance_to(500);
    CHECK(m2.now() == 1000);

    // Realised variance over many paths should match sigma^2 * open days,
    // plus one weekend gap carrying its own inflated variance.
    {
        const double sigma = 0.006;
        const double wmult = 1.30;
        double       s2    = 0.0;
        constexpr int kPaths = 4000;
        const SimTime horizon = kSecondsPerWeek;
        for (int i = 0; i < kPaths; ++i) {
            MarketConfig c;
            c.sigma_daily      = sigma;
            c.weekend_vol_mult = wmult;
            c.seed             = 1000 + static_cast<std::uint64_t>(i);
            Market mk(c);
            mk.advance_to(horizon);
            const double lr = std::log(to_rate(mk.mid()) / to_rate(c.mid0));
            s2 += lr * lr;
        }
        s2 /= kPaths;
        const double open_days = static_cast<double>(kSecondsPerWeek - kWeekendSec) / 86400.0;
        const double wknd_days = static_cast<double>(kWeekendSec) / 86400.0;
        const double expected  = sigma * sigma * open_days + sigma * wmult * sigma * wmult * wknd_days;
        CHECK_MSG(std::fabs(s2 / expected - 1.0) < 0.06,
                  "realised variance " + std::to_string(s2) + " vs expected " +
                      std::to_string(expected));
    }

    // The path lives on a fixed grid, so querying it in many small hops must
    // give bit-identical results to one big hop. This is what lets Phase 4 run
    // two policies against the same world.
    {
        MarketConfig c;
        c.sigma_daily = 0.005;
        c.seed        = 7001;
        Market one(c);
        one.advance_to(2 * kSecondsPerDay);

        Market many(c);
        for (SimTime t = 60; t <= 2 * kSecondsPerDay; t += 60) many.advance_to(t);
        CHECK(one.mid() == many.mid());

        Market erratic(c);
        erratic.advance_to(500);
        erratic.advance_to(90'000);
        erratic.advance_to(1'000);  // ignored
        erratic.advance_to(2 * kSecondsPerDay);
        CHECK(erratic.mid() == one.mid());
        CHECK(erratic.comp_bid() == one.comp_bid());

        // Reading ahead must not disturb the sequence either.
        Market peek(c);
        (void)peek.mid_at(5 * kSecondsPerDay);
        peek.advance_to(2 * kSecondsPerDay);
        CHECK(peek.mid() == one.mid());
    }

    // The competitor sheet is symmetric at the moment it refreshes, and then
    // goes stale: after the mid moves, the two sides are no longer equal.
    {
        MarketConfig c;
        c.sigma_daily = 0.0;
        Market mk(c);
        CHECK(vt::near(mk.comp_effective_half_bps(true), c.comp_half_bps, 0.05));
        CHECK(vt::near(mk.comp_effective_half_bps(false), c.comp_half_bps, 0.05));

        MarketConfig d;
        d.sigma_daily = 0.02;
        d.seed        = 44;
        Market mv(d);
        mv.advance_to(3 * 3600);  // move before the first refresh of the day
        const double bb = mv.comp_effective_half_bps(true);
        const double ss = mv.comp_effective_half_bps(false);
        CHECK(bb >= d.comp_min_eff_half_bps);
        CHECK(ss >= d.comp_min_eff_half_bps);
        CHECK(!vt::near(bb, ss, 1e-6));  // stale sheet is asymmetric
    }
}

// ---------------------------------------------------------------------------
void test_demand_curve() {
    vt::section("flow / demand curve and oracle");

    const double mu = std::log(200.0), s = 0.45;

    // Endpoints and monotonicity.
    CHECK(vt::near(accept_probability(mu, s, std::exp(mu)), 0.5, 1e-9));
    CHECK(accept_probability(mu, s, 1.0) > 0.999);
    CHECK(accept_probability(mu, s, 100'000.0) < 0.001);
    CHECK(vt::near(accept_probability(mu, s, -5.0), 1.0));
    double prev = 2.0;
    for (double d = 1.0; d < 2000.0; d *= 1.05) {
        const double p = accept_probability(mu, s, d);
        CHECK_MSG(p <= prev + 1e-12, "acceptance must be non-increasing in spread");
        prev = p;
    }

    // The analytic curve must match what the sampler actually produces.
    {
        Rng    g(4242);
        for (double delta : {80.0, 150.0, 200.0, 300.0, 450.0}) {
            long          hit = 0;
            constexpr int kN  = 60'000;
            for (int i = 0; i < kN; ++i)
                if (std::exp(mu + s * g.normal()) >= delta) ++hit;
            const double emp = static_cast<double>(hit) / kN;
            const double ana = accept_probability(mu, s, delta);
            CHECK_MSG(std::fabs(emp - ana) < 0.008,
                      "empirical " + std::to_string(emp) + " vs analytic " + std::to_string(ana) +
                          " at " + std::to_string(delta));
        }
    }

    // The oracle must beat a dense grid of alternatives, which is the whole
    // basis for reporting regret in Phase 4.
    for (double cost : {0.0, 1.5, 8.0}) {
        const double best  = oracle_optimal_half_bps(mu, s, cost);
        const double bestv = expected_margin_bps(mu, s, best, cost);
        int          worse = 0;
        for (int i = 0; i <= 4000; ++i) {
            const double d = std::exp(std::log(cost + 0.01) +
                                      (mu + 6.0 * s - std::log(cost + 0.01)) * i / 4000.0);
            if (expected_margin_bps(mu, s, d, cost) > bestv + 1e-9) ++worse;
        }
        CHECK_MSG(worse == 0, "grid beat the oracle in " + std::to_string(worse) +
                                  " places at cost " + std::to_string(cost));
        CHECK(best > cost);
    }

    // Structural properties of the optimum.
    CHECK(oracle_optimal_half_bps(mu, s, 8.0) > oracle_optimal_half_bps(mu, s, 0.0));
    CHECK(oracle_optimal_half_bps(std::log(400.0), s) >
          oracle_optimal_half_bps(std::log(200.0), s));
    // Scaling mu shifts the optimum proportionally.
    CHECK(vt::near(oracle_optimal_half_bps(std::log(400.0), s) /
                       oracle_optimal_half_bps(std::log(200.0), s),
                   2.0, 1e-6));
    // Zero margin is never optimal.
    CHECK(expected_margin_bps(mu, s, oracle_optimal_half_bps(mu, s)) > 0.0);
    CHECK(vt::near(expected_margin_bps(mu, s, 5.0, 10.0), 0.0));
}

// ---------------------------------------------------------------------------
void test_flow_generator() {
    vt::section("flow / arrivals and customers");

    FlowConfig fc = FlowConfig::defaults();
    CHECK_MSG(fc.validate().empty(), "defaults should validate: " + fc.validate());

    FlowConfig bad     = fc;
    bad.p_client_buys  = 1.5;
    CHECK(!bad.validate().empty());
    bad                    = fc;
    bad.tiers[0].sigma_ln  = 0.0;
    CHECK(!bad.validate().empty());
    bad                       = fc;
    bad.base_arrivals_per_day = 0.0;
    CHECK(!bad.validate().empty());

    FlowGenerator fg(fc);

    // The dominating rate really does dominate.
    for (SimTime t = 0; t < kSecondsPerWeek; t += 613)
        CHECK_MSG(fg.intensity(t) <= fg.max_intensity() + 1e-15,
                  "intensity exceeded lambda_max");

    // Retail demand does not stop when the interbank market shuts. That is the
    // asymmetry the weekend widener exists to price.
    CHECK(fg.intensity(kFridayCloseSec + 6 * 3600) > 0.0);
    CHECK(!is_open(kFridayCloseSec + 6 * 3600));

    // Thinning must reproduce the intended rate. Compare the realised count
    // over four weeks against the integrated intensity.
    {
        FlowGenerator g2(fc);
        const SimTime horizon = 4 * kSecondsPerWeek;
        long          n       = 0;
        SimTime       t       = 0;
        while (true) {
            t = g2.next_arrival(t);
            if (t >= horizon) break;
            ++n;
        }
        double integral = 0.0;
        for (SimTime u = 0; u < horizon; u += 60) integral += g2.intensity(u) * 60.0;
        CHECK_MSG(std::fabs(static_cast<double>(n) / integral - 1.0) < 0.04,
                  "arrival count " + std::to_string(n) + " vs expected " +
                      std::to_string(integral));
    }

    // Seasonality must actually be present. The previous version of this test
    // compared realised counts against the integral of intensity(), which is
    // self-consistent: deleting seasonality changed both sides equally and the
    // test still passed. These assertions pin the shape itself.
    {
        FlowGenerator g7(fc);
        // 14:00 IST is 08:30 UTC; 03:00 IST is 21:30 UTC the day before.
        const SimTime wed_peak  = 2 * kSecondsPerDay + static_cast<SimTime>(8.5 * 3600);
        const SimTime wed_night = 2 * kSecondsPerDay + static_cast<SimTime>(21.5 * 3600);
        const SimTime sat_peak  = 5 * kSecondsPerDay + static_cast<SimTime>(8.5 * 3600);
        CHECK(g7.intensity(wed_peak) > 4.0 * g7.intensity(wed_night));
        CHECK(g7.intensity(sat_peak) < g7.intensity(wed_peak));
        CHECK_MSG(std::fabs(g7.intensity(sat_peak) / g7.intensity(wed_peak) -
                            fc.day_factor[5] / fc.day_factor[2]) < 1e-9,
                  "weekday-to-Saturday ratio must follow day_factor");

        // And it must show up in the sampled arrivals, not just the intensity.
        FlowGenerator g8(fc);
        const SimTime horizon = 6 * kSecondsPerWeek;
        long          peak = 0, night = 0, weekday_n = 0, weekend_n = 0;
        SimTime       t = 0;
        while (true) {
            t = g8.next_arrival(t);
            if (t >= horizon) break;
            const MarketClock mc = clock_at(t);
            double h = mc.hour_utc + 5.5;
            if (h >= 24.0) h -= 24.0;
            if (h >= 12.0 && h < 16.0) ++peak;
            if (h >= 1.0 && h < 5.0) ++night;
            if (mc.weekday <= 4) ++weekday_n; else ++weekend_n;
        }
        CHECK(night > 0);
        CHECK_MSG(peak > 4 * night, "peak IST hours must dominate the small hours");
        // Five weekdays against two weekend days, with weekend traffic lighter
        // per day but not absent.
        CHECK(weekend_n > 0);
        CHECK(static_cast<double>(weekday_n) / 5.0 >
              1.3 * static_cast<double>(weekend_n) / 2.0);
    }

    // Arrivals must be strictly increasing.
    {
        FlowGenerator g3(fc);
        SimTime       t = 0, prev = -1;
        for (int i = 0; i < 5000; ++i) {
            t = g3.next_arrival(t);
            CHECK_MSG(t > prev, "arrivals must strictly increase");
            prev = t;
        }
    }

    // Customer draws: tier mix, side mix, size bounds, and the reservation
    // anchoring on the competitor.
    {
        FlowGenerator g4(fc);
        long          tiers[kTierCount]{};
        long          buys = 0;
        double        res_sum = 0.0;
        constexpr int kN      = 120'000;
        for (int i = 0; i < kN; ++i) {
            const Customer c = g4.make_customer(1000, 130.0, 130.0);
            ++tiers[static_cast<int>(c.tier)];
            if (c.client_buys) ++buys;
            const TierFlowParams& tp = fc.tiers[static_cast<int>(c.tier)];
            CHECK_MSG(c.size >= tp.size_min && c.size <= tp.size_max, "size out of bounds");
            CHECK_MSG(c.reservation_half_bps > 0.0, "reservation must be positive");
            if (c.tier == Tier::Retail && c.size == 5'000) res_sum += c.reservation_half_bps;
        }
        CHECK(vt::near(static_cast<double>(buys) / kN, fc.p_client_buys, 0.006));
        double wsum = 0.0;
        for (int i = 0; i < kTierCount; ++i) wsum += fc.tiers[i].arrival_weight;
        for (int i = 0; i < kTierCount; ++i)
            CHECK_MSG(std::fabs(static_cast<double>(tiers[i]) / kN -
                                fc.tiers[i].arrival_weight / wsum) < 0.008,
                      "tier mix off for " + std::string(tier_name(static_cast<Tier>(i))));
    }

    // Stickiness ordering: retail tolerates a wider markup than wealth. This
    // has to be measured from customers the generator actually produced, not
    // recomputed from the config, or the test proves nothing about the code.
    {
        FlowGenerator g5(fc);
        double        sum_anchor[kTierCount]{};
        long          n5[kTierCount]{};
        constexpr double kAnchor = 130.0;
        for (int i = 0; i < 400'000; ++i) {
            const Customer c = g5.make_customer(1000, kAnchor, kAnchor);
            // Strip the size term so what is left is ln(anchor * stickiness).
            const double size_term =
                fc.beta_size * std::log(static_cast<double>(c.size) /
                                        static_cast<double>(fc.size_ref));
            sum_anchor[static_cast<int>(c.tier)] += c.mu - size_term;
            ++n5[static_cast<int>(c.tier)];
        }
        double implied[kTierCount];
        for (int i = 0; i < kTierCount; ++i) {
            CHECK(n5[i] > 1000);
            implied[i] = std::exp(sum_anchor[i] / static_cast<double>(n5[i]));
            // The recovered stickiness must match the configured one.
            CHECK_MSG(std::fabs(implied[i] / (kAnchor * fc.tiers[i].stickiness) - 1.0) < 1e-9,
                      "implied stickiness wrong for " +
                          std::string(tier_name(static_cast<Tier>(i))));
        }
        CHECK(implied[0] > implied[1]);
        CHECK(implied[1] > implied[2]);
        CHECK(implied[2] > implied[3]);
        // Retail really does tolerate a markedly wider markup than wealth.
        CHECK(implied[0] > 1.4 * implied[3]);
    }

    // Bigger tickets shop harder: mu falls as size rises.
    {
        FlowConfig    fc2 = fc;
        fc2.tiers[0].size_log_sd = 1.6;  // widen so both bins are populated
        FlowGenerator g6(fc2);
        double small_mu = 0.0, big_mu = 0.0;
        long   ns = 0, nb = 0;
        for (int i = 0; i < 200'000; ++i) {
            const Customer c = g6.make_customer(1000, 130.0, 130.0);
            if (c.tier != Tier::Retail) continue;
            if (c.size < 1'000) { small_mu += c.mu; ++ns; }
            if (c.size > 20'000) { big_mu += c.mu; ++nb; }
        }
        CHECK(ns > 100 && nb > 100);
        if (ns > 0 && nb > 0)
            CHECK(small_mu / static_cast<double>(ns) > big_mu / static_cast<double>(nb));
    }
}

// ---------------------------------------------------------------------------
void test_desk() {
    vt::section("desk / accounting and hedging");

    const Points mid = 9'529'000;

    // A flat book with no trades has zero equity.
    Desk d0;
    CHECK(vt::near(d0.equity(mid), 0.0));
    CHECK(d0.position() == 0);

    // Client buys: desk sells, position goes short, cash comes in.
    Desk       d1;
    const Points ask = 9'720'000;
    d1.on_fill(true, 10'000, ask, mid);
    CHECK(d1.position() == -10'000);
    CHECK(d1.cash() > 0.0);
    // Equity equals the spread captured, exactly.
    CHECK(vt::near(d1.equity(mid), (to_rate(ask) - to_rate(mid)) * 10'000.0, 1e-6));
    CHECK(vt::near(d1.stats().gross_spread_pnl, d1.equity(mid), 1e-6));

    // Client sells: mirror image.
    Desk       d2;
    const Points bid = 9'330'000;
    d2.on_fill(false, 10'000, bid, mid);
    CHECK(d2.position() == 10'000);
    CHECK(d2.cash() < 0.0);
    CHECK(vt::near(d2.equity(mid), (to_rate(mid) - to_rate(bid)) * 10'000.0, 1e-6));

    // A round trip at the mid with no spread is exactly flat.
    Desk d3;
    d3.on_fill(true, 5'000, mid, mid);
    d3.on_fill(false, 5'000, mid, mid);
    CHECK(d3.position() == 0);
    CHECK(vt::near(d3.equity(mid), 0.0, 1e-6));

    // Zero and negative sizes are ignored.
    Desk d4;
    d4.on_fill(true, 0, ask, mid);
    d4.on_fill(true, -100, ask, mid);
    CHECK(d4.position() == 0 && d4.stats().trades == 0);

    // Hedging: inside the band nothing happens; outside it, back to flat.
    DeskConfig hc;
    hc.hedge_band     = 100'000;
    hc.hedge_cost_bps = 2.0;
    Desk d5(hc);
    d5.on_fill(false, 50'000, bid, mid);
    CHECK(d5.maybe_hedge(mid, 3600) == 0);
    CHECK(d5.position() == 50'000);

    Desk d6(hc);
    d6.on_fill(false, 150'000, bid, mid);
    const double before = d6.equity(mid);
    const Notional hq   = d6.maybe_hedge(mid, 3600);
    CHECK(hq == 150'000);
    CHECK(d6.position() == 0);
    CHECK(d6.stats().hedges == 1);
    CHECK(d6.stats().hedge_volume == 150'000);
    // Hedging costs exactly the configured slippage.
    const double expect_cost = to_rate(mid) * (2.0 / kBpsScale) * 150'000.0;
    CHECK(vt::near(d6.stats().hedge_cost, expect_cost, 1e-6));
    CHECK(vt::near(d6.equity(mid), before - expect_cost, 1e-6));

    // Short side hedges symmetrically.
    Desk d7(hc);
    d7.on_fill(true, 150'000, ask, mid);
    CHECK(d7.maybe_hedge(mid, 3600) == -150'000);
    CHECK(d7.position() == 0);
    CHECK(vt::near(d7.stats().hedge_cost, expect_cost, 1e-6));

    // A shut market cannot be hedged, however large the position.
    Desk d8(hc);
    d8.on_fill(false, 900'000, bid, mid);
    CHECK(d8.maybe_hedge(mid, kFridayCloseSec + 3600) == 0);
    CHECK(d8.position() == 900'000);
    CHECK(d8.maybe_hedge(mid, kSundayOpenSec) == 900'000);

    // Disabled hedging never trades.
    DeskConfig nc;
    nc.hedging_enabled = false;
    nc.hedge_band      = 1;
    Desk d9(nc);
    d9.on_fill(false, 500'000, bid, mid);
    CHECK(d9.maybe_hedge(mid, 3600) == 0);

    // Equity tracks the mid for an open position, and drawdown is recorded.
    Desk d10;
    d10.on_fill(false, 10'000, bid, mid);
    d10.mark(mid);
    const double e_at_mid = d10.equity(mid);
    const Points lower    = mid - 50'000;
    CHECK(d10.equity(lower) < e_at_mid);
    d10.mark(lower);
    CHECK(d10.stats().max_drawdown > 0.0);
    CHECK(vt::near(d10.stats().max_drawdown, e_at_mid - d10.equity(lower), 1e-6));
}

// ---------------------------------------------------------------------------
SimConfig small_sim(std::uint64_t seed = 5, SimTime weeks = 2) {
    SimConfig c;
    c.weeks         = weeks;
    c.market.seed   = seed;
    c.flow.seed     = seed * 7919 + 1;
    c.jitter_seed   = seed * 104729 + 2;
    return c;
}

void test_simulator() {
    vt::section("simulator / end to end");

    SimConfig cfg = small_sim();
    CHECK_MSG(cfg.validate().empty(), "default sim config should validate: " + cfg.validate());

    Simulator sim(cfg);
    sim.run();
    const SimSummary s = sim.summary();

    CHECK(s.events > 1000);
    CHECK(s.accepted > 0);
    CHECK(s.accepted < s.events);  // neither everyone nor nobody deals
    CHECK(s.hit_rate > 0.02 && s.hit_rate < 0.98);
    CHECK(sim.events().size() == sim.oracle().size());
    CHECK(s.client_volume > 0);

    // Event and oracle logs must stay aligned by id.
    bool aligned = true;
    for (std::size_t i = 0; i < sim.events().size(); ++i)
        if (sim.events()[i].id != sim.oracle()[i].id ||
            sim.events()[i].id != static_cast<std::int64_t>(i))
            aligned = false;
    CHECK(aligned);

    // Every logged quote must be internally consistent.
    int bad = 0;
    for (std::size_t i = 0; i < sim.events().size(); ++i) {
        const Event&       e = sim.events()[i];
        const OracleEvent& o = sim.oracle()[i];
        if (e.quoted_half_bps <= 0.0) ++bad;
        if (e.size <= 0) ++bad;
        if (e.mid <= 0 || e.quote_px <= 0) ++bad;
        if (e.comp_half_bps <= 0.0) ++bad;
        if (e.spread_mult <= 0.0) ++bad;
        // The label must be exactly the reservation comparison.
        if (e.accepted != (e.quoted_half_bps <= o.reservation_half_bps)) ++bad;
        // Buyers pay above the mid, sellers receive below it.
        if (e.client_buys && e.quote_px <= e.mid) ++bad;
        if (!e.client_buys && e.quote_px >= e.mid) ++bad;
        if (o.oracle_margin_bps < o.realised_margin_bps - 1e-9) ++bad;
        if (o.accept_prob < 0.0 || o.accept_prob > 1.0) ++bad;
    }
    CHECK_MSG(bad == 0, "inconsistent events: " + std::to_string(bad));

    // Regret is non-negative by construction: no policy beats the oracle.
    CHECK(s.regret_bps >= -1e-9);
    CHECK(s.mean_oracle_margin_bps > 0.0);

    // Events must be time-ordered and inside the horizon.
    bool ordered = true;
    for (std::size_t i = 1; i < sim.events().size(); ++i)
        if (sim.events()[i].t < sim.events()[i - 1].t) ordered = false;
    CHECK(ordered);
    CHECK(sim.events().back().t < cfg.weeks * kSecondsPerWeek);

    // Some flow must arrive while the interbank market is shut, otherwise the
    // weekend widener is never exercised.
    long closed = 0, weekend_wide = 0, intraday = 0;
    double closed_spread = 0.0, open_spread = 0.0;
    for (const Event& e : sim.events()) {
        if (!e.market_open) {
            ++closed;
            closed_spread += e.quoted_half_bps;
        } else if (e.horizon_h < 1.0) {
            ++intraday;
            open_spread += e.quoted_half_bps;
        }
        if (e.horizon_h > 24.0) ++weekend_wide;
    }
    CHECK(closed > 50);
    CHECK(weekend_wide > 50);
    CHECK(intraday > 50);
    // And quotes shown into the weekend really are wider.
    CHECK(closed_spread / static_cast<double>(closed) >
          open_spread / static_cast<double>(intraday));
}

void test_simulator_determinism() {
    vt::section("simulator / determinism and stream independence");

    Simulator a(small_sim(11)), b(small_sim(11));
    a.run();
    b.run();
    CHECK(a.events().size() == b.events().size());
    bool identical = a.events().size() == b.events().size();
    if (identical)
        for (std::size_t i = 0; i < a.events().size(); ++i) {
            const Event& x = a.events()[i];
            const Event& y = b.events()[i];
            if (x.t != y.t || x.size != y.size || x.mid != y.mid ||
                x.quote_px != y.quote_px || x.accepted != y.accepted) {
                identical = false;
                break;
            }
        }
    CHECK(identical);
    CHECK(vt::near(a.summary().total_pnl, b.summary().total_pnl, 1e-9));

    // A different seed gives a different run.
    Simulator c(small_sim(12));
    c.run();
    CHECK(!vt::near(a.summary().total_pnl, c.summary().total_pnl, 1e-6));

    // Independent streams: changing only the flow seed must leave the price
    // path untouched, so policy comparisons are not confounded by a new market.
    SimConfig p = small_sim(11);
    SimConfig q = small_sim(11);
    q.flow.seed += 1;
    Simulator sp(p), sq(q);
    sp.run();
    sq.run();
    CHECK(sp.summary().final_mid == sq.summary().final_mid);
    CHECK(sp.events().size() != sq.events().size() ||
          sp.events().front().size != sq.events().front().size);
}

void test_pnl_identity() {
    vt::section("simulator / pnl reconciles");

    // Rebuild the desk's position and cash from the event log alone and check
    // they match what the desk reports. If attribution and accounting ever
    // disagree, this fails.
    SimConfig cfg = small_sim(21);
    cfg.desk.hedging_enabled = false;  // no hedges, so the log is the whole story
    Simulator sim(cfg);
    sim.run();

    Notional pos  = 0;
    double   cash = 0.0, gross = 0.0;
    Notional vol  = 0;
    long     dealt = 0;
    for (const Event& e : sim.events()) {
        if (!e.accepted) continue;
        ++dealt;
        vol += e.size;
        const double px = to_rate(e.quote_px);
        const double m  = to_rate(e.mid);
        if (e.client_buys) {
            cash += px * static_cast<double>(e.size);
            pos -= e.size;
        } else {
            cash -= px * static_cast<double>(e.size);
            pos += e.size;
        }
        gross += std::fabs(px - m) * static_cast<double>(e.size);
    }

    const SimSummary s = sim.summary();
    CHECK(pos == s.final_position);
    CHECK(dealt == s.accepted);
    CHECK(vol == s.client_volume);
    CHECK(vt::near(gross, s.gross_spread_pnl, 1e-6));

    const double equity = cash + static_cast<double>(pos) * to_rate(s.final_mid);
    CHECK(vt::near(equity, s.total_pnl, 1e-6));
    // With no hedging, the attribution buckets must add back to the total.
    CHECK(vt::near(s.hedge_cost, 0.0));
    CHECK(vt::near(s.gross_spread_pnl + s.inventory_pnl, s.total_pnl, 1e-6));

    // The inventory the pricer was shown must be the position before that fill.
    Notional running = 0;
    int      mismatch = 0;
    for (const Event& e : sim.events()) {
        if (e.inventory != running) ++mismatch;
        if (e.accepted) running += e.client_buys ? -e.size : e.size;
    }
    CHECK_MSG(mismatch == 0, "inventory in the log drifted in " + std::to_string(mismatch) +
                                 " events");
}

void test_economics() {
    vt::section("simulator / economic behaviour");

    // Hedging costs money but cuts risk. Both must show up.
    SimConfig hedged = small_sim(31);
    SimConfig naked  = small_sim(31);
    naked.desk.hedging_enabled = false;
    Simulator sh(hedged), sn(naked);
    sh.run();
    sn.run();
    CHECK(sh.summary().hedges > 0);
    CHECK(sn.summary().hedges == 0);
    CHECK(sh.summary().max_abs_position < sn.summary().max_abs_position);
    CHECK(sh.summary().hedge_cost > 0.0);

    // A cheaper competitor drags customer willingness to pay down, so the desk
    // deals less at any given spread.
    SimConfig rich = small_sim(41);
    SimConfig lean = small_sim(41);
    lean.market.comp_half_bps = 60.0;
    Simulator sr(rich), sl(lean);
    sr.run();
    sl.run();
    CHECK(sl.summary().hit_rate < sr.summary().hit_rate);
    CHECK(sl.summary().mean_oracle_half_bps < sr.summary().mean_oracle_half_bps);

    // More exploration jitter must widen the spread of quotes actually shown,
    // which is what identifies the demand curve in Phase 3.
    SimConfig quiet = small_sim(51);
    quiet.jitter_log_sd = 0.02;
    SimConfig loud = small_sim(51);
    loud.jitter_log_sd = 0.35;
    Simulator sq2(quiet), sl2(loud);
    sq2.run();
    sl2.run();
    auto spread_sd = [](const Simulator& s) {
        double m = 0.0, m2 = 0.0;
        long   n = 0;
        for (const Event& e : s.events()) {
            if (e.tier != Tier::Retail || e.clamped) continue;
            m += e.quoted_half_bps;
            m2 += e.quoted_half_bps * e.quoted_half_bps;
            ++n;
        }
        if (n < 2) return 0.0;
        const double dn   = static_cast<double>(n);
        const double mean = m / dn;
        return std::sqrt(std::max(0.0, m2 / dn - mean * mean));
    };
    CHECK(spread_sd(sl2) > 2.0 * spread_sd(sq2));

    // Zero jitter means every multiplier is exactly one.
    SimConfig none = small_sim(61);
    none.jitter_log_sd = 0.0;
    Simulator sz(none);
    sz.run();
    bool all_one = true;
    for (const Event& e : sz.events())
        if (!vt::near(e.spread_mult, 1.0)) all_one = false;
    CHECK(all_one);

    // Bad configs are rejected rather than silently simulated.
    bool threw = false;
    try {
        SimConfig broken = small_sim(71);
        broken.weeks     = 0;
        Simulator bad(broken);
        bad.run();
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    test_rng();
    test_sim_calendar();
    test_market_process();
    test_demand_curve();
    test_flow_generator();
    test_desk();
    test_simulator();
    test_simulator_determinism();
    test_pnl_identity();
    test_economics();
    return vt::report("simulation");
}
