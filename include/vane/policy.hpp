// Vane - retail FX auto-quoter
// policy.hpp : how a spread gets chosen.
//
// Phase 1's Pricer computes a spread from a tier table. Phase 3 learned a
// demand curve and picked the spread that maximises expected margin. Phase 4
// needs both to run inside the same simulator, on the same price path, so the
// difference between them is attributable to the policy and nothing else.
//
// Hence this interface. `StaticPolicy` is the incumbent, unchanged. The
// learned policy loads coefficients fitted in Python and evaluates them here,
// which is also how such a model is actually deployed: fitted offline in a
// research stack, served from the low-latency path.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vane/pricer.hpp"

namespace vane {

// Everything a policy needs to choose a spread. A superset of QuoteRequest,
// since the learned policy also conditions on the competitor's effective
// spread and on the time of day.
struct PolicyContext {
    QuoteRequest req;
    double       comp_half_bps = 0.0;
    double       hour_utc      = 0.0;
    bool         market_open   = true;
    bool         client_buys   = true;
};

class QuotePolicy {
public:
    virtual ~QuotePolicy() = default;

    // Returns the half-spread in bps the policy wants, before the engine's
    // clamps and before exploration jitter.
    virtual double half_spread_bps(const PolicyContext& ctx) const = 0;

    virtual std::string name() const = 0;
};

// The incumbent: whatever the Phase 1 pricing engine produces.
class StaticPolicy : public QuotePolicy {
public:
    explicit StaticPolicy(const Pricer* pricer) noexcept : pricer_(pricer) {}

    double      half_spread_bps(const PolicyContext& ctx) const override;
    std::string name() const override { return "static"; }

private:
    const Pricer* pricer_;
};

// ---------------------------------------------------------------------------
// Learned policy
// ---------------------------------------------------------------------------

// A logistic demand model exported from Phase 3: P(accept) = sigmoid(w . z),
// where z is the standardised feature vector. Feature order is fixed by
// `kFeatureNames` and asserted against the file on load, so a reordering in
// the Python layer fails loudly instead of silently mispricing.
struct LogisticDemandParams {
    std::vector<std::string> feature_names;
    std::vector<double>      mean;
    std::vector<double>      scale;
    std::vector<double>      coef;
    double                   intercept = 0.0;

    // Per-tier spread range the training data actually covered. The policy
    // will not price outside it -- the central lesson of Phase 3.
    double support_lo[kTierCount]{};
    double support_hi[kTierCount]{};

    static LogisticDemandParams load(const std::string& path);
    std::string                 validate() const;
};

// The feature vector, in the order the Python layer emits it.
extern const char* const kFeatureNames[];
extern const int         kFeatureCount;

class LearnedPolicy : public QuotePolicy {
public:
    LearnedPolicy(LogisticDemandParams params, double cost_bps, bool respect_support = true);

    double      half_spread_bps(const PolicyContext& ctx) const override;
    std::string name() const override {
        return respect_support_ ? "learned" : "learned-unconstrained";
    }

    // Exposed for testing: P(accept) at a given spread.
    double accept_probability(const PolicyContext& ctx, double half_bps) const;

private:
    void build_features(const PolicyContext& ctx, double half_bps, double* out) const;

    LogisticDemandParams params_;
    double               cost_bps_;
    bool                 respect_support_;
};

}  // namespace vane
