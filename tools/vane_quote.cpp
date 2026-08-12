// vane-quote : inspect the pricing core from the command line.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vane/calendar.hpp"
#include "vane/pricer.hpp"

using namespace vane;

namespace {

void usage() {
    std::puts(
        "vane-quote - retail FX auto-quoter (Phase 1 pricing core)\n"
        "\n"
        "  vane-quote [options]           print one quote\n"
        "  vane-quote --sweep-inventory   spread/skew across a position range\n"
        "  vane-quote --sweep-week        the weekend widener, hour by hour\n"
        "  vane-quote --sweep-tier        every tier, side by side\n"
        "  vane-quote --bench             quote latency\n"
        "\n"
        "options:\n"
        "  --mid <rate>        interbank mid            (default 95.29)\n"
        "  --inv <units>       net base-ccy inventory   (default 0)\n"
        "  --size <units>      ticket size              (default 5000)\n"
        "  --tier <name>       retail|corporate|private|wealth\n"
        "  --sigma <frac>      daily volatility         (default 0.004)\n"
        "  --horizon <hours>   hedge horizon            (default 0.5)\n"
        "  --weekday <0-6>     Mon=0; derives horizon from the calendar\n"
        "  --hour <0-24>       UTC hour; used with --weekday\n");
}

double arg_d(const char* v) { return std::atof(v); }

void print_quote(const QuoteRequest& r, const Quote& q) {
    std::printf("mid          %.5f\n", to_rate(r.mid));
    std::printf("tier         %s\n", std::string(tier_name(r.tier)).c_str());
    std::printf("inventory    %lld\n", static_cast<long long>(r.inventory));
    std::printf("size         %lld\n", static_cast<long long>(r.size));
    std::printf("sigma/day    %.2f bps\n", r.sigma_daily * kBpsScale);
    std::printf("horizon      %.2f h  ->  sigma_h %.2f bps\n", r.hedge_horizon_hours,
                q.sigma_horizon_bps);
    std::puts("");
    if (q.reject != RejectReason::None) {
        std::printf("REJECTED     %s\n", std::string(reject_name(q.reject)).c_str());
        return;
    }
    std::printf("reservation  %.5f   (skew %+.2f bps)\n", to_rate(q.reservation), q.skew_bps);
    std::printf("half-spread  %.2f bps%s%s%s\n", q.half_spread_bps,
                q.clamped_by_floor ? "  [floor]" : "", q.clamped_by_ceiling ? "  [ceiling]" : "",
                q.clamped_by_rival ? "  [rival]" : "");
    std::puts("");
    if (q.bid_valid)
        std::printf("BID          %.5f\n", to_rate(q.bid));
    else
        std::puts("BID          -- suppressed (position limit)");
    if (q.ask_valid)
        std::printf("ASK          %.5f\n", to_rate(q.ask));
    else
        std::puts("ASK          -- suppressed (position limit)");
    if (q.two_sided())
        std::printf("spread       %.5f  (%.1f bps)\n", to_rate(q.spread()),
                    2.0 * q.half_spread_bps);
}

void sweep_inventory(Pricer& p, QuoteRequest r) {
    r.hedge_horizon_hours = 24.0;
    std::printf("inventory sweep  (mid %.4f, tier %s, sigma %.0f bps/day, horizon %.1fh)\n\n",
                to_rate(r.mid), std::string(tier_name(r.tier)).c_str(),
                r.sigma_daily * kBpsScale, r.hedge_horizon_hours);
    std::puts("   inventory      skew bps      bid          ask       spread bps");
    std::puts("  ------------------------------------------------------------------");
    for (Notional inv = -3'000'000; inv <= 3'000'000; inv += 500'000) {
        r.inventory   = inv;
        const Quote q = p.quote(r);
        std::printf("  %+11lld   %+9.2f   %10.5f   %10.5f   %8.1f%s\n",
                    static_cast<long long>(inv), q.skew_bps, to_rate(q.bid), to_rate(q.ask),
                    2.0 * q.half_spread_bps, q.two_sided() ? "" : "   [one-sided]");
    }
}

void sweep_week(Pricer& p, QuoteRequest r) {
    std::printf("weekend widener  (mid %.4f, tier %s, sigma %.0f bps/day)\n\n", to_rate(r.mid),
                std::string(tier_name(r.tier)).c_str(), r.sigma_daily * kBpsScale);
    std::puts("   when            horizon h    sigma_h bps   spread bps      bid          ask");
    std::puts("  ---------------------------------------------------------------------------");
    const char* days[7] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
    struct P { int d; double h; };
    const std::vector<P> pts = {{2, 12.0}, {4, 8.0},  {4, 14.0}, {4, 16.0}, {4, 18.0},
                                {4, 20.0}, {4, 21.5}, {5, 6.0},  {6, 12.0}, {6, 22.5}};
    for (const P& pt : pts) {
        r.hedge_horizon_hours = hedge_horizon_hours({pt.d, pt.h});
        const Quote q         = p.quote(r);
        char        when[32];
        std::snprintf(when, sizeof(when), "%s %05.2f UTC", days[pt.d], pt.h);
        std::printf("  %-14s  %8.2f   %10.2f   %9.1f   %10.5f   %10.5f\n", when,
                    r.hedge_horizon_hours, q.sigma_horizon_bps, 2.0 * q.half_spread_bps,
                    to_rate(q.bid), to_rate(q.ask));
    }
}

void sweep_tier(Pricer& p, QuoteRequest r) {
    std::printf("tier sweep  (mid %.4f, size %lld, sigma %.0f bps/day, horizon %.1fh)\n\n",
                to_rate(r.mid), static_cast<long long>(r.size), r.sigma_daily * kBpsScale,
                r.hedge_horizon_hours);
    std::puts("   tier          spread bps       bid          ask      client cost/1k");
    std::puts("  -------------------------------------------------------------------");
    for (int t = 0; t < kTierCount; ++t) {
        r.tier        = static_cast<Tier>(t);
        const Quote q = p.quote(r);
        const double cost = 1000.0 * (to_rate(q.ask) - to_rate(r.mid));
        std::printf("  %-12s   %9.1f   %10.5f   %10.5f   %10.2f\n",
                    std::string(tier_name(r.tier)).c_str(), 2.0 * q.half_spread_bps,
                    to_rate(q.bid), to_rate(q.ask), cost);
    }
    std::puts("\n  client cost/1k = extra INR paid on a 1,000-unit purchase vs the interbank mid");
}

void bench(Pricer& p, QuoteRequest r) {
    constexpr int kWarm = 100'000;
    constexpr int kRuns = 5'000'000;
    volatile std::int64_t sink = 0;

    for (int i = 0; i < kWarm; ++i) {
        r.inventory = i % 1'000'000;
        sink += p.quote(r).bid;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kRuns; ++i) {
        r.inventory = i % 1'000'000;
        sink += p.quote(r).bid;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const auto   elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const double ns      = static_cast<double>(elapsed) / static_cast<double>(kRuns);
    std::printf("quote(): %.1f ns/call   (%.2f M quotes/s)\n", ns, 1000.0 / ns);
    (void)sink;
}

}  // namespace

int main(int argc, char** argv) {
    QuoteRequest r;
    r.mid                 = round_to_points(95.29);
    r.size                = 5'000;
    r.sigma_daily         = 0.0040;
    r.hedge_horizon_hours = 0.5;

    const char* mode    = "quote";
    int         weekday = -1;
    double      hour    = 12.0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--sweep-inventory") mode = "sweep-inventory";
        else if (a == "--sweep-week")      mode = "sweep-week";
        else if (a == "--sweep-tier")      mode = "sweep-tier";
        else if (a == "--bench")           mode = "bench";
        else if (a == "--mid")     r.mid = round_to_points(arg_d(next("--mid")));
        else if (a == "--inv")     r.inventory = std::atoll(next("--inv"));
        else if (a == "--size")    r.size = std::atoll(next("--size"));
        else if (a == "--sigma")   r.sigma_daily = arg_d(next("--sigma"));
        else if (a == "--horizon") r.hedge_horizon_hours = arg_d(next("--horizon"));
        else if (a == "--weekday") weekday = std::atoi(next("--weekday"));
        else if (a == "--hour")    hour = arg_d(next("--hour"));
        else if (a == "--tier") {
            if (!parse_tier(next("--tier"), r.tier)) {
                std::fprintf(stderr, "unknown tier\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage();
            return 2;
        }
    }

    if (weekday >= 0 && weekday <= 6) {
        r.hedge_horizon_hours = hedge_horizon_hours({weekday, hour});
    }

    PricingConfig cfg = PricingConfig::defaults();
    if (const std::string err = cfg.validate(); !err.empty()) {
        std::fprintf(stderr, "bad config: %s\n", err.c_str());
        return 1;
    }
    Pricer p(cfg);

    if (std::strcmp(mode, "sweep-inventory") == 0)      sweep_inventory(p, r);
    else if (std::strcmp(mode, "sweep-week") == 0)      sweep_week(p, r);
    else if (std::strcmp(mode, "sweep-tier") == 0)      sweep_tier(p, r);
    else if (std::strcmp(mode, "bench") == 0)           bench(p, r);
    else                                                print_quote(r, p.quote(r));
    return 0;
}
