#include <string>

#include "harness.hpp"
#include "vane/calendar.hpp"
#include "vane/pricer.hpp"

using namespace vane;

namespace {

QuoteRequest base_req() {
    QuoteRequest r;
    r.mid                 = 9'529'000;  // USD/INR 95.29, per the appendix snapshot
    r.inventory           = 0;
    r.size                = 5'000;
    r.tier                = Tier::Retail;
    r.sigma_daily         = 0.0040;  // 40 bps/day
    r.hedge_horizon_hours = 0.5;
    r.tick_age_ms         = 0;
    return r;
}

void test_types() {
    vt::section("types / fixed point");

    CHECK(to_rate(9'529'000) == 95.29);
    CHECK(round_to_points(95.29) == 9'529'000);

    // Directional rounding must never shrink the desk's margin.
    CHECK(floor_to_points(95.294999) == 9'529'499);
    CHECK(ceil_to_points(95.290001) == 9'529'001);
    CHECK(floor_to_points(95.29) == 9'529'000);
    CHECK(ceil_to_points(95.29) == 9'529'000);

    Tier t;
    CHECK(parse_tier("retail", t) && t == Tier::Retail);
    CHECK(parse_tier("wealth", t) && t == Tier::Wealth);
    CHECK(!parse_tier("platinum", t));
    CHECK(tier_name(Tier::Private) == "private");
    CHECK(reject_name(RejectReason::StaleTick) == "stale_tick");
}

void test_config() {
    vt::section("config validation");

    PricingConfig c = PricingConfig::defaults();
    CHECK_MSG(c.validate().empty(), "defaults should validate: " + c.validate());

    // Tier bands must be ordered cheapest for the best segment.
    CHECK(c.tiers[0].base_half_bps > c.tiers[1].base_half_bps);
    CHECK(c.tiers[1].base_half_bps > c.tiers[2].base_half_bps);
    CHECK(c.tiers[2].base_half_bps > c.tiers[3].base_half_bps);

    PricingConfig bad = c;
    bad.tiers[0].ceil_half_bps = 10.0;
    CHECK(!bad.validate().empty());

    bad                        = c;
    bad.tiers[2].base_half_bps = 1.0;  // below its own floor
    CHECK(!bad.validate().empty());

    bad                = c;
    bad.cost_floor_bps = 500.0;  // above every ceiling
    CHECK(!bad.validate().empty());

    bad                = c;
    bad.risk_aversion  = -1.0;
    CHECK(!bad.validate().empty());

    bad                  = c;
    bad.inventory_limit  = 0;
    CHECK(!bad.validate().empty());

    bad         = c;
    bad.min_mid = 0;
    CHECK(!bad.validate().empty());
}

void test_calendar() {
    vt::section("calendar / hedge horizon");

    CHECK(market_open({0, 9.0}));    // Monday morning
    CHECK(market_open({4, 21.0}));   // Friday, an hour before the close
    CHECK(!market_open({4, 22.0}));  // Friday close is exclusive
    CHECK(!market_open({5, 12.0}));  // Saturday
    CHECK(!market_open({6, 21.0}));  // Sunday, before the open
    CHECK(market_open({6, 22.0}));   // Sunday open
    CHECK(market_open({6, 23.5}));

    CHECK(vt::near(hours_to_close({4, 20.0}), 2.0));
    CHECK(vt::near(hours_to_close({0, 22.0}), 96.0));
    CHECK(vt::near(hours_to_close({6, 22.0}), 120.0));  // fresh week from the open
    CHECK(vt::near(hours_to_close({5, 10.0}), 0.0));    // closed
    CHECK(vt::near(hours_to_open({5, 22.0}), 24.0));
    CHECK(vt::near(hours_to_open({6, 12.0}), 10.0));
    CHECK(vt::near(hours_to_open({1, 12.0}), 0.0));     // open

    // Mid-week the horizon is just the intraday hold.
    CHECK(vt::near(hedge_horizon_hours({2, 12.0}, 0.5, 8.0), 0.5));

    // Into the Friday close the weekend gap phases in.
    const double h_far  = hedge_horizon_hours({4, 12.0}, 0.5, 8.0);  // 10h to close
    const double h_near = hedge_horizon_hours({4, 18.0}, 0.5, 8.0);  // 4h to close
    const double h_shut = hedge_horizon_hours({4, 21.9}, 0.5, 8.0);
    CHECK(vt::near(h_far, 0.5));
    CHECK(h_near > h_far);
    CHECK(h_shut > h_near);
    CHECK(h_shut < 0.5 + kWeekendGapHours + 1e-9);

    // Over the weekend itself the horizon is the time until the market reopens.
    CHECK(vt::near(hedge_horizon_hours({5, 22.0}, 0.5, 8.0), 24.5));

    // Monotone: horizon never decreases as Friday wears on.
    double prev = -1.0;
    for (double hr = 0.0; hr < 22.0; hr += 0.25) {
        const double h = hedge_horizon_hours({4, hr}, 0.5, 8.0);
        CHECK_MSG(h >= prev - 1e-12, "horizon must be monotone through Friday");
        prev = h;
    }
}

void test_basic_quote() {
    vt::section("pricer / basic quote");

    Pricer      p;
    QuoteRequest r = base_req();
    Quote        q = p.quote(r);

    CHECK(q.reject == RejectReason::None);
    CHECK(q.two_sided());
    CHECK(q.bid < q.ask);
    CHECK(q.bid < r.mid);
    CHECK(q.ask > r.mid);
    CHECK(q.spread() > 0);

    // Flat book, so no skew and the reservation price sits on the mid.
    CHECK(vt::near(q.skew_bps, 0.0));
    CHECK(q.reservation == r.mid);

    // Retail base is 200 bps plus a small vol term; nowhere near the ceiling.
    CHECK(q.half_spread_bps > 200.0);
    CHECK(q.half_spread_bps < 400.0);
    CHECK(!q.clamped_by_ceiling);
    CHECK(!q.clamped_by_floor);
    CHECK(!q.clamped_by_rival);

    // The realised half-spread should match the reported one to within a point.
    const double realised_bid_bps = (to_rate(r.mid) - to_rate(q.bid)) / to_rate(r.mid) * kBpsScale;
    CHECK(std::fabs(realised_bid_bps - q.half_spread_bps) < 0.02);

    // Exact rounding direction: the bid is floored and the ask ceilinged, so a
    // swapped or symmetric rounding rule is caught at single-point resolution.
    for (int t = 0; t < kTierCount; ++t) {
        QuoteRequest rr = base_req();
        rr.tier         = static_cast<Tier>(t);
        rr.mid          = 9'529'137;  // deliberately not a round number
        const Quote qq  = p.quote(rr);
        const double res  = to_rate(qq.reservation);
        const double half = qq.half_spread_bps / kBpsScale;
        CHECK(qq.bid == floor_to_points(res * (1.0 - half)));
        CHECK(qq.ask == ceil_to_points(res * (1.0 + half)));
    }
}

void test_guardrails() {
    vt::section("pricer / guardrails");

    Pricer p;

    QuoteRequest r = base_req();
    r.mid          = 0;
    CHECK(p.quote(r).reject == RejectReason::InvalidInput);

    r     = base_req();
    r.mid = -1;
    CHECK(p.quote(r).reject == RejectReason::InvalidInput);

    // Too small to carry a spread in fixed point.
    r     = base_req();
    r.mid = p.config().min_mid - 1;
    CHECK(p.quote(r).reject == RejectReason::InvalidInput);
    r.mid = p.config().min_mid;
    CHECK(p.quote(r).reject == RejectReason::None);

    r             = base_req();
    r.sigma_daily = -0.01;
    CHECK(p.quote(r).reject == RejectReason::InvalidInput);

    r      = base_req();
    r.size = -1;
    CHECK(p.quote(r).reject == RejectReason::InvalidInput);

    r                     = base_req();
    r.hedge_horizon_hours = std::nan("");
    CHECK(p.quote(r).reject == RejectReason::InvalidInput);

    r             = base_req();
    r.tick_age_ms = 2'001;
    CHECK(p.quote(r).reject == RejectReason::StaleTick);
    CHECK(!p.quote(r).ok());

    r             = base_req();
    r.tick_age_ms = 2'000;  // boundary is inclusive
    CHECK(p.quote(r).reject == RejectReason::None);

    r      = base_req();
    r.size = 250'001;
    CHECK(p.quote(r).reject == RejectReason::SizeAboveLimit);

    r      = base_req();
    r.size = 250'000;
    CHECK(p.quote(r).reject == RejectReason::None);
}

void test_inventory_skew() {
    vt::section("pricer / inventory skew");

    Pricer      p;
    QuoteRequest flat = base_req();
    flat.hedge_horizon_hours = 24.0;  // give the skew something to bite on

    QuoteRequest lng = flat;
    lng.inventory    = 1'500'000;
    QuoteRequest shrt = flat;
    shrt.inventory    = -1'500'000;

    const Quote qf = p.quote(flat);
    const Quote ql = p.quote(lng);
    const Quote qs = p.quote(shrt);

    // Long inventory shades the whole price down; short shades it up.
    CHECK(ql.skew_bps < 0.0);
    CHECK(qs.skew_bps > 0.0);
    CHECK(ql.bid < qf.bid);
    CHECK(ql.ask < qf.ask);
    CHECK(qs.bid > qf.bid);
    CHECK(qs.ask > qf.ask);
    CHECK(vt::near(ql.skew_bps, -qs.skew_bps, 1e-9));

    // Skew moves the price, it does not widen it. The half-spread in bps is
    // untouched; the absolute spread in points therefore scales with the
    // reservation price, which is the intended behaviour rather than a leak.
    CHECK(vt::near(ql.half_spread_bps, qf.half_spread_bps, 1e-9));
    const double ratio_spread = double(ql.spread()) / double(qf.spread());
    const double ratio_price  = double(ql.reservation) / double(qf.reservation);
    CHECK(vt::near(ratio_spread, ratio_price, 1e-5));

    // The cap binds for an extreme position.
    QuoteRequest huge = flat;
    huge.inventory    = 500'000'000;
    const Quote qh    = p.quote(huge);
    CHECK(vt::near(std::fabs(qh.skew_bps), p.config().max_skew_bps, 1e-9));
}

void test_vol_and_weekend() {
    vt::section("pricer / volatility and weekend widening");

    Pricer p;

    QuoteRequest calm  = base_req();
    QuoteRequest rough = base_req();
    rough.sigma_daily  = 0.0120;
    CHECK(p.quote(rough).half_spread_bps > p.quote(calm).half_spread_bps);

    // Same volatility, but priced against a weekend hold instead of a
    // half-hour one. This is the report's weekend cushion, arrived at from the
    // horizon rather than as a bolt-on rule.
    QuoteRequest intraday = base_req();
    intraday.hedge_horizon_hours = hedge_horizon_hours({2, 12.0});
    QuoteRequest friday          = base_req();
    friday.hedge_horizon_hours   = hedge_horizon_hours({4, 21.9});

    const Quote qi = p.quote(intraday);
    const Quote qf = p.quote(friday);
    CHECK(qf.sigma_horizon_bps > qi.sigma_horizon_bps * 5.0);
    CHECK(qf.half_spread_bps > qi.half_spread_bps);
    CHECK(qf.spread() > qi.spread());

    // sqrt-of-time: quadrupling the horizon should double horizon vol.
    QuoteRequest h1 = base_req();
    h1.hedge_horizon_hours = 6.0;
    QuoteRequest h4 = base_req();
    h4.hedge_horizon_hours = 24.0;
    CHECK(vt::near(p.quote(h4).sigma_horizon_bps, 2.0 * p.quote(h1).sigma_horizon_bps, 1e-9));

    // Zero horizon means no volatility component at all.
    QuoteRequest h0 = base_req();
    h0.hedge_horizon_hours = 0.0;
    const Quote q0         = p.quote(h0);
    CHECK(vt::near(q0.sigma_horizon_bps, 0.0));
    CHECK(vt::near(q0.half_spread_bps, p.config().tiers[0].base_half_bps, 1e-9));
}

void test_tiers_and_size() {
    vt::section("pricer / tiers and ticket size");

    Pricer p;
    double prev = 1e9;
    for (int i = 0; i < kTierCount; ++i) {
        QuoteRequest r = base_req();
        r.tier         = static_cast<Tier>(i);
        const Quote q  = p.quote(r);
        CHECK_MSG(q.half_spread_bps < prev,
                  std::string("tier ") + std::string(tier_name(r.tier)) +
                      " should price inside the previous tier");
        prev = q.half_spread_bps;
    }

    // Wealth flow is quoted materially tighter than retail flow.
    QuoteRequest rr = base_req();
    QuoteRequest rw = base_req();
    rw.tier         = Tier::Wealth;
    CHECK(p.quote(rw).spread() * 4 < p.quote(rr).spread());

    // Bigger tickets widen, sub-reference tickets do not get a discount.
    QuoteRequest small = base_req();
    small.size         = 500;
    QuoteRequest ref = base_req();
    ref.size         = 5'000;
    QuoteRequest big = base_req();
    big.size         = 200'000;
    big.tier         = Tier::Wealth;  // avoid the retail ceiling masking the effect
    QuoteRequest ref_w = ref;
    ref_w.tier         = Tier::Wealth;

    CHECK(vt::near(p.quote(small).half_spread_bps, p.quote(ref).half_spread_bps, 1e-9));
    CHECK(p.quote(big).half_spread_bps > p.quote(ref_w).half_spread_bps);
}

void test_clamps() {
    vt::section("pricer / floor and ceiling clamps");

    Pricer p;

    // Extreme volatility drives the raw spread past the tier ceiling.
    QuoteRequest wild  = base_req();
    wild.tier          = Tier::Wealth;
    wild.sigma_daily   = 0.05;
    wild.hedge_horizon_hours = 48.0;
    const Quote qw     = p.quote(wild);
    CHECK(qw.clamped_by_ceiling);
    CHECK(vt::near(qw.half_spread_bps, p.config().tiers[3].ceil_half_bps, 1e-9));

    // A tier whose base sits under the economic floor is lifted back to it.
    PricingConfig cfg      = PricingConfig::defaults();
    cfg.tiers[3].base_half_bps  = 12.0;
    cfg.tiers[3].floor_half_bps = 12.0;
    cfg.cost_floor_bps          = 12.0;
    CHECK(cfg.validate().empty());

    Pricer       p2;
    PricingConfig cheap        = cfg;
    cheap.tiers[3].base_half_bps  = 13.0;
    cheap.tiers[3].floor_half_bps = 13.0;
    cheap.cost_floor_bps          = 30.0;
    CHECK(cheap.validate().empty());
    Pricer       p3(cheap);
    QuoteRequest r = base_req();
    r.tier         = Tier::Wealth;
    r.sigma_daily  = 0.0;
    const Quote q3 = p3.quote(r);
    CHECK(q3.clamped_by_floor);
    CHECK(vt::near(q3.half_spread_bps, 30.0, 1e-9));
    (void)p2;
}

void test_competitor_clamp() {
    vt::section("pricer / competitor clamp");

    PricingConfig cfg              = PricingConfig::defaults();
    cfg.competitive_clamp_enabled  = true;
    cfg.competitive_max_slip_bps   = 30.0;
    Pricer p(cfg);
    Pricer plain;  // clamp disabled

    QuoteRequest r = base_req();
    // Thomas Cook's snapshot: buying at 93.79, selling at 96.28.
    r.competitor_bid = 9'379'000;
    r.competitor_ask = 9'628'000;

    const Quote base    = plain.quote(r);
    const Quote clamped = p.quote(r);

    CHECK(clamped.clamped_by_rival);
    CHECK(clamped.bid > base.bid);  // forced to bid up towards the rival
    CHECK(clamped.ask < base.ask);  // and to offer lower
    CHECK(clamped.spread() < base.spread());

    // The clamp must never price through the economic floor.
    const double floor_off = cfg.cost_floor_bps / kBpsScale;
    const double res       = to_rate(clamped.reservation);
    CHECK(to_rate(clamped.bid) <= res * (1.0 - floor_off) + 1e-9);
    CHECK(to_rate(clamped.ask) >= res * (1.0 + floor_off) - 1e-9);

    // An uncompetitive rival leaves the quote untouched.
    QuoteRequest weak = base_req();
    weak.competitor_bid = 9'000'000;
    weak.competitor_ask = 9'900'000;
    const Quote qweak   = p.quote(weak);
    CHECK(!qweak.clamped_by_rival);
    CHECK(qweak.bid == base.bid);
    CHECK(qweak.ask == base.ask);

    // Zero means "unknown" and must be ignored rather than treated as a price.
    QuoteRequest none = base_req();
    const Quote  qn   = p.quote(none);
    CHECK(!qn.clamped_by_rival);
    CHECK(qn.bid == base.bid && qn.ask == base.ask);
}

void test_position_limits() {
    vt::section("pricer / position limits and one-sided quotes");

    Pricer p;
    const Notional hard = p.config().inventory_hard_limit;

    // Nearly at the long limit: the desk stops bidding but keeps offering.
    QuoteRequest r = base_req();
    r.size         = 100'000;
    r.inventory    = hard - 50'000;
    Quote q        = p.quote(r);
    CHECK(!q.bid_valid);
    CHECK(q.ask_valid);
    CHECK(q.ok());
    CHECK(q.reject == RejectReason::None);

    // Mirror image at the short limit.
    r.inventory = -(hard - 50'000);
    q           = p.quote(r);
    CHECK(q.bid_valid);
    CHECK(!q.ask_valid);

    // A ticket larger than the whole limit blocks both sides.
    r.size      = 250'000;
    r.inventory = 0;
    PricingConfig tight        = PricingConfig::defaults();
    tight.inventory_hard_limit = 100'000;
    Pricer      p2(tight);
    const Quote q2 = p2.quote(r);
    CHECK(!q2.ok());
    CHECK(q2.reject == RejectReason::InventoryHardLimit);

    // Exactly at the limit is still allowed.
    r.size      = 100'000;
    r.inventory = 0;
    CHECK(p2.quote(r).two_sided());
}

void test_determinism() {
    vt::section("pricer / determinism");

    Pricer      p;
    QuoteRequest r = base_req();
    r.inventory    = 750'000;
    r.sigma_daily  = 0.0063;
    r.hedge_horizon_hours = 17.25;

    const Quote a = p.quote(r);
    for (int i = 0; i < 1000; ++i) {
        const Quote b = p.quote(r);
        if (b.bid != a.bid || b.ask != a.ask || b.reservation != a.reservation) {
            CHECK_MSG(false, "quote is not deterministic");
            return;
        }
    }
    CHECK(true);
}

}  // namespace

int main() {
    test_types();
    test_config();
    test_calendar();
    test_basic_quote();
    test_guardrails();
    test_inventory_skew();
    test_vol_and_weekend();
    test_tiers_and_size();
    test_clamps();
    test_competitor_clamp();
    test_position_limits();
    test_determinism();
    return vt::report("unit");
}
