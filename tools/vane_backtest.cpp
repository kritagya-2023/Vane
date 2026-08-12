// vane-backtest : run competing quote policies against the same world.
//
// Every comparison here uses common random numbers. The market seed, the flow
// seed and the jitter seed are held fixed across policies, and the price path
// is generated on a fixed grid that does not depend on when it is queried, so
// two policies see the identical sequence of prices and the identical sequence
// of customers. Whatever difference appears in the PnL is the policy.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vane/policy.hpp"
#include "vane/simulator.hpp"

using namespace vane;

namespace {

void usage() {
    std::puts(
        "vane-backtest - compare quote policies on identical price paths (Phase 4)\n"
        "\n"
        "  vane-backtest --model models/logistic.txt [options]\n"
        "\n"
        "options:\n"
        "  --model <path>     exported logistic demand model\n"
        "  --weeks <n>        weeks per run              (default 12)\n"
        "  --seeds <n>        independent worlds         (default 5)\n"
        "  --jitter <sd>      exploration jitter         (default 0.15)\n"
        "  --sigma <frac>     daily volatility           (default 0.004)\n"
        "  --comp <bps>       competitor half-spread     (default 130)\n"
        "  --no-support       let the learned policy price outside its support\n"
        "  --ablate           run the risk-control ablations\n"
        "  --csv <path>       write the per-seed table\n");
}

struct Row {
    std::string policy;
    int         seed = 0;
    SimSummary  s;
};

SimConfig make_cfg(std::uint64_t seed, SimTime weeks, double jitter, double sigma, double comp) {
    SimConfig c;
    c.weeks                 = weeks;
    c.jitter_log_sd         = jitter;
    c.market.sigma_daily    = sigma;
    c.market.comp_half_bps  = comp;
    c.market.seed           = seed;
    c.flow.seed             = seed * 7919 + 1;
    c.jitter_seed           = seed * 104729 + 2;
    return c;
}

Row run(const std::string& label, SimConfig cfg, std::shared_ptr<QuotePolicy> pol, int seed) {
    cfg.policy = std::move(pol);
    Simulator sim(cfg);
    sim.run();
    Row r;
    r.policy = label;
    r.seed   = seed;
    r.s      = sim.summary();
    return r;
}

void print_table(const std::vector<Row>& rows) {
    std::puts("\n  policy                seed   quotes    hit%   spread   margin   regret"
              "        total pnl      maxpos   hedges");
    std::puts("  ------------------------------------------------------------------------"
              "---------------------------------");
    for (const Row& r : rows) {
        std::printf("  %-20s %5d %8lld  %6.1f %8.1f %8.2f %8.2f %16.0f %11lld %8lld\n",
                    r.policy.c_str(), r.seed, static_cast<long long>(r.s.events),
                    100.0 * r.s.hit_rate, r.s.mean_quoted_half_bps,
                    r.s.mean_realised_margin_bps, r.s.regret_bps, r.s.total_pnl,
                    static_cast<long long>(r.s.max_abs_position),
                    static_cast<long long>(r.s.hedges));
    }
}

struct Agg {
    int    n = 0;
    double pnl = 0, pnl2 = 0, regret = 0, margin = 0, spread = 0, hit = 0, maxpos = 0;
    double volume = 0;
};

void print_summary(const std::vector<Row>& rows) {
    std::vector<std::string> order;
    std::vector<Agg>         aggs;
    for (const Row& r : rows) {
        std::size_t i = 0;
        for (; i < order.size(); ++i)
            if (order[i] == r.policy) break;
        if (i == order.size()) {
            order.push_back(r.policy);
            aggs.emplace_back();
        }
        Agg& a = aggs[i];
        ++a.n;
        a.pnl += r.s.total_pnl;
        a.pnl2 += r.s.total_pnl * r.s.total_pnl;
        a.regret += r.s.regret_bps;
        a.margin += r.s.mean_realised_margin_bps;
        a.spread += r.s.mean_quoted_half_bps;
        a.hit += r.s.hit_rate;
        a.maxpos += static_cast<double>(r.s.max_abs_position);
        a.volume += static_cast<double>(r.s.client_volume);
    }

    std::puts("\n  mean across seeds");
    std::puts("  policy                 spread     hit%   margin   regret       mean pnl"
              "        pnl sd     max pos");
    std::puts("  ----------------------------------------------------------------------"
              "-------------------------------");
    double base_pnl = 0.0;
    for (std::size_t i = 0; i < order.size(); ++i) {
        const Agg&   a = aggs[i];
        const double n = a.n;
        const double mean = a.pnl / n;
        const double var  = std::max(0.0, a.pnl2 / n - mean * mean);
        if (i == 0) base_pnl = mean;
        std::printf("  %-20s %8.1f %8.1f %8.2f %8.2f %14.0f %13.0f %11.0f\n", order[i].c_str(),
                    a.spread / n, 100.0 * a.hit / n, a.margin / n, a.regret / n, mean,
                    std::sqrt(var), a.maxpos / n);
    }
    if (order.size() > 1 && base_pnl != 0.0) {
        std::puts("\n  versus the static baseline");
        for (std::size_t i = 1; i < order.size(); ++i) {
            const double mean = aggs[i].pnl / aggs[i].n;
            std::printf("    %-20s %+14.0f  (%+.2f%%)\n", order[i].c_str(), mean - base_pnl,
                        100.0 * (mean - base_pnl) / std::fabs(base_pnl));
        }
    }
}

void write_csv(const std::string& path, const std::vector<Row>& rows) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    std::fprintf(f, "policy,seed,events,hit_rate,mean_spread_bps,mean_margin_bps,regret_bps,"
                    "total_pnl,gross_spread,hedge_cost,inventory_pnl,client_volume,"
                    "max_abs_position,hedges,max_drawdown\n");
    for (const Row& r : rows) {
        std::fprintf(f, "%s,%d,%lld,%.6f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%lld,%lld,%lld,%.2f\n",
                     r.policy.c_str(), r.seed, static_cast<long long>(r.s.events), r.s.hit_rate,
                     r.s.mean_quoted_half_bps, r.s.mean_realised_margin_bps, r.s.regret_bps,
                     r.s.total_pnl, r.s.gross_spread_pnl, r.s.hedge_cost, r.s.inventory_pnl,
                     static_cast<long long>(r.s.client_volume),
                     static_cast<long long>(r.s.max_abs_position),
                     static_cast<long long>(r.s.hedges), r.s.max_drawdown);
    }
    std::fclose(f);
    std::printf("\nwrote %s\n", path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, csv_path;
    SimTime     weeks   = 12;
    int         seeds   = 5;
    double      jitter  = 0.15, sigma = 0.004, comp = 130.0;
    bool        no_support = false, ablate = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* w) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", w); std::exit(2); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--model")   model_path = next("--model");
        else if (a == "--weeks")   weeks = std::atoll(next("--weeks"));
        else if (a == "--seeds")   seeds = std::atoi(next("--seeds"));
        else if (a == "--jitter")  jitter = std::atof(next("--jitter"));
        else if (a == "--sigma")   sigma = std::atof(next("--sigma"));
        else if (a == "--comp")    comp = std::atof(next("--comp"));
        else if (a == "--csv")     csv_path = next("--csv");
        else if (a == "--no-support") no_support = true;
        else if (a == "--ablate")  ablate = true;
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(); return 2; }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "--model is required (see analysis/export_model.py)\n");
        return 2;
    }

    try {
        const LogisticDemandParams params = LogisticDemandParams::load(model_path);
        std::printf("loaded %s\n", model_path.c_str());
        for (int t = 0; t < kTierCount; ++t)
            std::printf("  support %-10s %7.1f - %7.1f bps\n",
                        std::string(tier_name(static_cast<Tier>(t))).c_str(),
                        params.support_lo[t], params.support_hi[t]);
        std::printf("\n%d seeds x %lld weeks, jitter %.2f, sigma %.4f, competitor %.0f bps\n",
                    seeds, static_cast<long long>(weeks), jitter, sigma, comp);

        const double cost = SimConfig{}.desk.hedge_cost_bps;
        std::vector<Row> rows;

        for (int seed = 1; seed <= seeds; ++seed) {
            const SimConfig base = make_cfg(static_cast<std::uint64_t>(seed), weeks, jitter,
                                            sigma, comp);
            rows.push_back(run("static", base, nullptr, seed));
            rows.push_back(run("learned", base,
                               std::make_shared<LearnedPolicy>(params, cost, true), seed));
            if (no_support)
                rows.push_back(run("learned-no-support", base,
                                   std::make_shared<LearnedPolicy>(params, cost, false), seed));

            if (ablate) {
                // Turn off one risk control at a time, so its contribution is
                // visible rather than assumed.
                SimConfig no_skew = base;
                no_skew.pricing.risk_aversion = 0.0;
                rows.push_back(run("learned-no-skew", no_skew,
                                   std::make_shared<LearnedPolicy>(params, cost, true), seed));

                SimConfig no_wknd = base;
                no_wknd.intraday_horizon_hours = 0.5;
                no_wknd.pricing.vol_coeff      = 0.0;  // no volatility term at all
                rows.push_back(run("learned-no-volterm", no_wknd,
                                   std::make_shared<LearnedPolicy>(params, cost, true), seed));

                SimConfig no_hedge = base;
                no_hedge.desk.hedging_enabled = false;
                rows.push_back(run("learned-no-hedge", no_hedge,
                                   std::make_shared<LearnedPolicy>(params, cost, true), seed));
            }
        }

        print_table(rows);
        print_summary(rows);
        if (!csv_path.empty()) write_csv(csv_path, rows);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
