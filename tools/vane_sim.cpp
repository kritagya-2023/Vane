// vane-sim : run the flow simulation and export the labelled dataset.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>

#include "vane/policy.hpp"
#include "vane/simulator.hpp"

using namespace vane;

namespace {

void usage() {
    std::puts(
        "vane-sim - customer flow simulator (Phase 2)\n"
        "\n"
        "  vane-sim [options]\n"
        "\n"
        "options:\n"
        "  --weeks <n>        weeks to simulate           (default 4)\n"
        "  --seed <n>         master seed                 (default 1)\n"
        "  --sigma <frac>     daily volatility            (default 0.004)\n"
        "  --comp <bps>       competitor half-spread      (default 130)\n"
        "  --jitter <sd>      exploration jitter, log sd  (default 0.18)\n"
        "  --band <units>     hedge band                  (default 400000)\n"
        "  --no-hedge         disable hedging\n"
        "  --arrivals <n>     base arrivals per day       (default 520)\n"
        "  --model <path>     drive quoting with an exported learned policy\n"
        "  --out <prefix>     write <prefix>_events.csv and <prefix>_oracle.csv\n"
        "  --by-tier          print the per-tier breakdown\n"
        "  --demand-curve     print realised acceptance against quoted spread\n");
}

void print_summary(const SimSummary& s, const SimConfig& cfg) {
    std::printf("simulation: %lld weeks, seed %llu, policy %s\n\n",
                static_cast<long long>(cfg.weeks),
                static_cast<unsigned long long>(cfg.market.seed), s.policy_name.c_str());

    std::puts("flow");
    std::printf("  quotes shown          %lld\n", static_cast<long long>(s.events));
    std::printf("  dealt                 %lld  (hit rate %.1f%%)\n",
                static_cast<long long>(s.accepted), 100.0 * s.hit_rate);
    std::printf("  client volume         %.2f M base ccy\n",
                static_cast<double>(s.client_volume) / 1e6);

    std::puts("\npricing");
    std::printf("  mean quoted half      %.1f bps\n", s.mean_quoted_half_bps);
    std::printf("  mean oracle half      %.1f bps\n", s.mean_oracle_half_bps);
    std::printf("  mean margin realised  %.2f bps/unit\n", s.mean_realised_margin_bps);
    std::printf("  mean margin oracle    %.2f bps/unit\n", s.mean_oracle_margin_bps);
    std::printf("  regret vs oracle      %.2f bps/unit  (%.1f%% of attainable)\n", s.regret_bps,
                s.mean_oracle_margin_bps > 0.0
                    ? 100.0 * s.regret_bps / s.mean_oracle_margin_bps
                    : 0.0);

    std::puts("\npnl (quote currency)");
    std::printf("  gross spread          %+15.2f\n", s.gross_spread_pnl);
    std::printf("  hedge cost            %+15.2f\n", -s.hedge_cost);
    std::printf("  inventory / market    %+15.2f\n", s.inventory_pnl);
    std::printf("  ------------------------------------\n");
    std::printf("  total                 %+15.2f\n", s.total_pnl);

    std::puts("\nrisk");
    std::printf("  hedges                %lld\n", static_cast<long long>(s.hedges));
    std::printf("  max abs position      %lld\n", static_cast<long long>(s.max_abs_position));
    std::printf("  final position        %lld\n", static_cast<long long>(s.final_position));
    std::printf("  max drawdown          %.2f\n", s.max_drawdown);
    std::printf("  final mid             %.5f  (from %.5f)\n", to_rate(s.final_mid),
                to_rate(cfg.market.mid0));
}

void print_by_tier(const Simulator& sim) {
    struct Agg {
        long long n = 0, dealt = 0;
        double    quoted = 0.0, oracle = 0.0, margin = 0.0, omargin = 0.0;
        long long volume = 0;
    };
    Agg a[kTierCount];
    const auto& ev = sim.events();
    const auto& or_ = sim.oracle();
    for (std::size_t i = 0; i < ev.size(); ++i) {
        Agg& g = a[static_cast<int>(ev[i].tier)];
        ++g.n;
        if (ev[i].accepted) {
            ++g.dealt;
            g.volume += ev[i].size;
        }
        g.quoted += ev[i].quoted_half_bps;
        g.oracle += or_[i].oracle_half_bps;
        g.margin += or_[i].realised_margin_bps;
        g.omargin += or_[i].oracle_margin_bps;
    }
    std::puts("\nby tier");
    std::puts("  tier         quotes    hit%   quoted   oracle   margin   best   regret   volume");
    std::puts("  -------------------------------------------------------------------------------");
    for (int i = 0; i < kTierCount; ++i) {
        const Agg& g = a[i];
        if (g.n == 0) continue;
        const double n = static_cast<double>(g.n);
        std::printf("  %-11s %7lld  %5.1f  %7.1f  %7.1f  %7.2f %6.2f  %7.2f  %7.2fM\n",
                    std::string(tier_name(static_cast<Tier>(i))).c_str(), g.n,
                    100.0 * static_cast<double>(g.dealt) / n, g.quoted / n, g.oracle / n,
                    g.margin / n, g.omargin / n, (g.omargin - g.margin) / n,
                    static_cast<double>(g.volume) / 1e6);
    }
    std::puts("\n  quoted/oracle in bps; margin = expected bps per unit notional");
}

void print_demand_curve(const Simulator& sim) {
    // Realised acceptance rate against the quoted spread, which is the shape
    // Phase 3 has to recover. Retail only, so the tier mix does not blur it.
    constexpr int kBins = 12;
    long long     n[kBins]{}, k[kBins]{};
    double        lo = 100.0, hi = 700.0;
    for (const Event& e : sim.events()) {
        if (e.tier != Tier::Retail) continue;
        int b = static_cast<int>((e.quoted_half_bps - lo) / (hi - lo) * kBins);
        if (b < 0) b = 0;
        if (b >= kBins) b = kBins - 1;
        ++n[b];
        if (e.accepted) ++k[b];
    }
    std::puts("\nrealised demand curve (retail)");
    std::puts("  quoted half-spread      quotes    accepted");
    std::puts("  ---------------------------------------------------------");
    for (int b = 0; b < kBins; ++b) {
        if (n[b] == 0) continue;
        const double a  = lo + (hi - lo) * b / kBins;
        const double bb = lo + (hi - lo) * (b + 1) / kBins;
        const double p  = static_cast<double>(k[b]) / static_cast<double>(n[b]);
        std::printf("  %5.0f - %5.0f bps   %8lld   %6.1f%%  ", a, bb, n[b], 100.0 * p);
        for (int i = 0; i < static_cast<int>(p * 40.0 + 0.5); ++i) std::putchar('#');
        std::putchar('\n');
    }
}

}  // namespace

int main(int argc, char** argv) {
    SimConfig cfg;
    std::string   out, model_path;
    bool          by_tier = false, demand = false;
    std::uint64_t seed    = 1;

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
        else if (a == "--weeks")    cfg.weeks = std::atoll(next("--weeks"));
        else if (a == "--seed")     seed = std::strtoull(next("--seed"), nullptr, 10);
        else if (a == "--sigma")    cfg.market.sigma_daily = std::atof(next("--sigma"));
        else if (a == "--comp")     cfg.market.comp_half_bps = std::atof(next("--comp"));
        else if (a == "--jitter")   cfg.jitter_log_sd = std::atof(next("--jitter"));
        else if (a == "--band")     cfg.desk.hedge_band = std::atoll(next("--band"));
        else if (a == "--no-hedge") cfg.desk.hedging_enabled = false;
        else if (a == "--arrivals") cfg.flow.base_arrivals_per_day = std::atof(next("--arrivals"));
        else if (a == "--out")      out = next("--out");
        else if (a == "--model")    model_path = next("--model");
        else if (a == "--by-tier")  by_tier = true;
        else if (a == "--demand-curve") demand = true;
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage();
            return 2;
        }
    }

    // Independent streams: changing the flow seed leaves the price path alone,
    // which makes A/B comparisons across policies far less noisy.
    cfg.market.seed = seed;
    cfg.flow.seed   = seed * 7919 + 1;
    cfg.jitter_seed = seed * 104729 + 2;

    try {
        if (!model_path.empty()) {
            cfg.policy = std::make_shared<LearnedPolicy>(
                LogisticDemandParams::load(model_path), cfg.desk.hedge_cost_bps, true);
        }
        Simulator sim(cfg);
        sim.run();
        print_summary(sim.summary(), cfg);
        if (by_tier) print_by_tier(sim);
        if (demand) print_demand_curve(sim);
        if (!out.empty()) {
            sim.write_events_csv(out + "_events.csv");
            sim.write_oracle_csv(out + "_oracle.csv");
            std::printf("\nwrote %s_events.csv and %s_oracle.csv\n", out.c_str(), out.c_str());
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
