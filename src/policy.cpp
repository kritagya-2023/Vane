#include "vane/policy.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vane {

// Must match analysis/vane_data.py FEATURES exactly, in order.
const char* const kFeatureNames[] = {
    "log_spread",  "log_comp",   "log_size",        "log_horizon",
    "sigma_daily", "abs_inventory", "is_buy",       "hour_sin",
    "hour_cos",    "is_weekend_flow",
    "tier_retail", "tier_corporate", "tier_private", "tier_wealth",
};
const int kFeatureCount = 14;

namespace {
constexpr double kSizeRef = 5000.0;

double sigmoid(double z) noexcept {
    if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
    const double e = std::exp(z);
    return e / (1.0 + e);
}
}  // namespace

// ---------------------------------------------------------------------------
double StaticPolicy::half_spread_bps(const PolicyContext& ctx) const {
    QuoteRequest r     = ctx.req;
    r.spread_multiplier = 1.0;
    return pricer_->quote(r).half_spread_bps;
}

// ---------------------------------------------------------------------------
std::string LogisticDemandParams::validate() const {
    const std::size_t n = static_cast<std::size_t>(kFeatureCount);
    if (feature_names.size() != n) return "feature count mismatch";
    for (std::size_t i = 0; i < n; ++i) {
        if (feature_names[i] != kFeatureNames[i]) {
            return "feature order mismatch at position " + std::to_string(i) + ": file has '" +
                   feature_names[i] + "', engine expects '" + kFeatureNames[i] + "'";
        }
    }
    if (mean.size() != n || scale.size() != n || coef.size() != n)
        return "parameter vectors have inconsistent length";
    for (std::size_t i = 0; i < n; ++i)
        if (!(scale[i] > 0.0)) return "non-positive scale for " + feature_names[i];

    // Demand must slope down in spread, or the argmax is unbounded.
    if (coef[0] >= 0.0) return "log_spread coefficient is non-negative: demand slopes upward";

    for (int t = 0; t < kTierCount; ++t) {
        if (!(support_lo[t] > 0.0) || support_hi[t] < support_lo[t])
            return "bad support bounds for tier " + std::string(tier_name(static_cast<Tier>(t)));
    }
    return {};
}

LogisticDemandParams LogisticDemandParams::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open model file: " + path);

    LogisticDemandParams p;
    std::string          line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string        key;
        ss >> key;
        if (key == "intercept") {
            ss >> p.intercept;
        } else if (key == "feature") {
            std::string name;
            double      mean = 0, scale = 0, coef = 0;
            ss >> name >> mean >> scale >> coef;
            p.feature_names.push_back(name);
            p.mean.push_back(mean);
            p.scale.push_back(scale);
            p.coef.push_back(coef);
        } else if (key == "support") {
            std::string tname;
            double      lo = 0, hi = 0;
            ss >> tname >> lo >> hi;
            Tier t;
            if (!parse_tier(tname, t)) throw std::runtime_error("unknown tier: " + tname);
            p.support_lo[static_cast<int>(t)] = lo;
            p.support_hi[static_cast<int>(t)] = hi;
        }
    }
    if (const std::string e = p.validate(); !e.empty())
        throw std::runtime_error("invalid model file " + path + ": " + e);
    return p;
}

// ---------------------------------------------------------------------------
LearnedPolicy::LearnedPolicy(LogisticDemandParams params, double cost_bps, bool respect_support)
    : params_(std::move(params)), cost_bps_(cost_bps), respect_support_(respect_support) {}

void LearnedPolicy::build_features(const PolicyContext& ctx, double half_bps, double* out) const {
    const QuoteRequest& r = ctx.req;
    const double hour = ctx.hour_utc;

    out[0]  = std::log(half_bps);
    out[1]  = std::log(std::max(ctx.comp_half_bps, 1e-9));
    out[2]  = std::log(static_cast<double>(r.size) / kSizeRef);
    out[3]  = std::log(std::max(r.hedge_horizon_hours, 0.05));
    out[4]  = r.sigma_daily;
    out[5]  = std::fabs(static_cast<double>(r.inventory)) / 1e6;
    out[6]  = ctx.client_buys ? 1.0 : 0.0;
    out[7]  = std::sin(2.0 * M_PI * hour / 24.0);
    out[8]  = std::cos(2.0 * M_PI * hour / 24.0);
    out[9]  = ctx.market_open ? 0.0 : 1.0;
    out[10] = r.tier == Tier::Retail ? 1.0 : 0.0;
    out[11] = r.tier == Tier::Corporate ? 1.0 : 0.0;
    out[12] = r.tier == Tier::Private ? 1.0 : 0.0;
    out[13] = r.tier == Tier::Wealth ? 1.0 : 0.0;
}

double LearnedPolicy::accept_probability(const PolicyContext& ctx, double half_bps) const {
    double x[16];
    build_features(ctx, half_bps, x);
    double z = params_.intercept;
    for (int i = 0; i < kFeatureCount; ++i)
        z += params_.coef[i] * (x[i] - params_.mean[i]) / params_.scale[i];
    return sigmoid(z);
}

double LearnedPolicy::half_spread_bps(const PolicyContext& ctx) const {
    const int t = static_cast<int>(ctx.req.tier);

    double lo = std::max(cost_bps_ * 1.01, 12.0);
    double hi = 600.0;
    if (respect_support_) {
        // The Phase 3 lesson, enforced in the serving path rather than only in
        // the research notebook: never price where the training data was
        // silent.
        lo = std::max(lo, params_.support_lo[t]);
        hi = std::min(hi, params_.support_hi[t]);
    }
    if (hi <= lo) return lo;

    // The objective is (delta - cost) * P(accept), an increasing linear term
    // times a decreasing probability, so it has a single interior peak. Grid
    // then refine.
    auto value = [&](double d) { return (d - cost_bps_) * accept_probability(ctx, d); };

    constexpr int kGrid = 64;
    double        best_d = lo, best_v = value(lo);
    for (int i = 1; i <= kGrid; ++i) {
        const double d = std::exp(std::log(lo) + (std::log(hi) - std::log(lo)) * i / kGrid);
        const double v = value(d);
        if (v > best_v) { best_v = v; best_d = d; }
    }

    double width = (std::log(hi) - std::log(lo)) / kGrid;
    for (int pass = 0; pass < 3; ++pass) {
        const double a = std::max(std::log(best_d) - width, std::log(lo));
        const double b = std::min(std::log(best_d) + width, std::log(hi));
        for (int i = 0; i <= 10; ++i) {
            const double d = std::exp(a + (b - a) * i / 10.0);
            const double v = value(d);
            if (v > best_v) { best_v = v; best_d = d; }
        }
        width /= 4.0;
    }
    return best_d;
}

}  // namespace vane
