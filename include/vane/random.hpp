// Vane - retail FX auto-quoter
// random.hpp : deterministic pseudo-random generation.
//
// Self-contained rather than <random>, because libstdc++ and libc++ disagree on
// the sequence their distributions produce from the same engine. A simulation
// that reports different PnL on a different machine is not reproducible, so the
// samplers are written out here.
#pragma once

#include <cmath>
#include <cstdint>

namespace vane {

class Rng {
public:
    explicit Rng(std::uint64_t seed) noexcept { reseed(seed); }

    void reseed(std::uint64_t seed) noexcept {
        // splitmix64 to spread a small seed across the whole state.
        std::uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < 4; ++i) {
            std::uint64_t x = (z += 0x9E3779B97F4A7C15ULL);
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            s_[i] = x ^ (x >> 31);
        }
        has_cached_normal_ = false;
    }

    // xoshiro256**
    std::uint64_t next_u64() noexcept {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t      = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Half-open [0, 1). Never returns exactly 0 or 1, so logs are always safe.
    double uniform() noexcept {
        const double u = static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
        return u <= 0.0 ? 0x1.0p-53 : u;
    }

    double uniform(double lo, double hi) noexcept { return lo + uniform() * (hi - lo); }

    bool bernoulli(double p) noexcept { return uniform() < p; }

    // Box-Muller, caching the second variate.
    double normal() noexcept {
        if (has_cached_normal_) {
            has_cached_normal_ = false;
            return cached_normal_;
        }
        const double u1 = uniform();
        const double u2 = uniform();
        const double r  = std::sqrt(-2.0 * std::log(u1));
        const double th = 6.283185307179586476925286766559 * u2;
        cached_normal_     = r * std::sin(th);
        has_cached_normal_ = true;
        return r * std::cos(th);
    }

    double normal(double mean, double sd) noexcept { return mean + sd * normal(); }

    double lognormal(double log_mean, double log_sd) noexcept {
        return std::exp(log_mean + log_sd * normal());
    }

    double exponential(double rate) noexcept { return -std::log(uniform()) / rate; }

    // Categorical draw over unnormalised weights.
    int categorical(const double* weights, int n) noexcept {
        double total = 0.0;
        for (int i = 0; i < n; ++i) total += weights[i];
        double u   = uniform() * total;
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
            acc += weights[i];
            if (u < acc) return i;
        }
        return n - 1;
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    std::uint64_t s_[4]{};
    double        cached_normal_     = 0.0;
    bool          has_cached_normal_ = false;
};

// --- normal distribution helpers -------------------------------------------
inline double norm_cdf(double z) noexcept {
    return 0.5 * std::erfc(-z * 0.7071067811865475244);
}

inline double norm_pdf(double z) noexcept {
    return 0.3989422804014326779 * std::exp(-0.5 * z * z);
}

}  // namespace vane
