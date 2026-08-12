// Vane - retail FX auto-quoter
// types.hpp : fixed-point rate representation and core enums.
#pragma once

#include <cmath>
#include <cstdint>
#include <string_view>

namespace vane {

// ---------------------------------------------------------------------------
// Rates are held as integers, never as doubles, so that a quote is bit-exact
// and reproducible across machines. One "point" is 1e-5 of quote currency per
// unit of base currency. USD/INR 95.29000 -> 9529000 points.
// ---------------------------------------------------------------------------
using Points   = std::int64_t;  // fixed-point exchange rate
using Notional = std::int64_t;  // base-currency units (whole USD)

inline constexpr std::int64_t kPointScale = 100000;
inline constexpr double       kBpsScale   = 10000.0;

constexpr double to_rate(Points p) noexcept {
    return static_cast<double>(p) / static_cast<double>(kPointScale);
}

constexpr Points from_rate_exact(double r) noexcept {
    return static_cast<Points>(r * static_cast<double>(kPointScale));
}

// Directional rounding. The bank's bid is rounded down and its ask is rounded
// up, so that fixed-point truncation can never eat into the quoted margin.
inline Points floor_to_points(double r) noexcept {
    return static_cast<Points>(std::floor(r * static_cast<double>(kPointScale)));
}
inline Points ceil_to_points(double r) noexcept {
    return static_cast<Points>(std::ceil(r * static_cast<double>(kPointScale)));
}
inline Points round_to_points(double r) noexcept {
    return static_cast<Points>(std::llround(r * static_cast<double>(kPointScale)));
}

// ---------------------------------------------------------------------------
// Client segmentation. Ordered cheapest-last: Retail pays the widest markup,
// Wealth the tightest. Phase 3 replaces the hand-set tier base with a spread
// chosen by the learned fill-probability model.
// ---------------------------------------------------------------------------
enum class Tier : std::uint8_t {
    Retail    = 0,
    Corporate = 1,
    Private   = 2,
    Wealth    = 3,
};
inline constexpr int kTierCount = 4;

std::string_view tier_name(Tier t) noexcept;
bool             parse_tier(std::string_view s, Tier& out) noexcept;

enum class RejectReason : std::uint8_t {
    None = 0,
    InvalidInput,       // malformed request
    StaleTick,          // upstream mid is older than the configured tolerance
    SizeAboveLimit,     // ticket exceeds the desk's single-trade cap
    InventoryHardLimit, // both sides would breach the position limit
};

std::string_view reject_name(RejectReason r) noexcept;

}  // namespace vane
