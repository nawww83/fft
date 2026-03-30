#pragma once

#include <bit>
#include <cstdint>
#include <concepts>
#include <cassert>

namespace utils {

/**
 * @brief Константный реверс байт (для static_assert)
 */
template<typename T>
constexpr T internal_byteswap(T v) noexcept {
    if constexpr (sizeof(T) == 1) return v;
    if constexpr (sizeof(T) == 2) return static_cast<T>((v << 8) | (v >> 8));
    if constexpr (sizeof(T) == 4) {
        return static_cast<T>(((v << 24) & 0xFF000000u) | ((v << 8) & 0x00FF0000u) |
                             ((v >> 8) & 0x0000FF00u) | ((v >> 24) & 0x000000FFu));
    }
    if constexpr (sizeof(T) == 8) {
        uint64_t x = static_cast<uint64_t>(v);
        x = ((x & 0x00FF00FF00FF00FFull) << 8) | ((x & 0xFF00FF00FF00FF00ull) >> 8);
        x = ((x & 0x0000FFFF0000FFFFull) << 16) | ((x & 0xFFFF0000FFFF0000ull) >> 16);
        return static_cast<T>((x << 32) | (x >> 32));
    }
    return v;
}

/**
 * @brief Быстрый реверс бит (C++23).
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr T fast_bit_reverse(T val, int bits) noexcept {
    static_assert(sizeof(T) <= 8);
    
    if (bits <= 0) return 0;
    assert(bits <= static_cast<int>(sizeof(T) * 8));

    T v = val;

    // Маски как константы T для подавления ворнингов MSVC
    const T mask1 = static_cast<T>(0x5555555555555555ull);
    const T mask2 = static_cast<T>(0xAAAAAAAAAAAAAAAAull);
    const T mask3 = static_cast<T>(0x3333333333333333ull);
    const T mask4 = static_cast<T>(0xCCCCCCCCCCCCCCCCull);
    const T mask5 = static_cast<T>(0x0F0F0F0F0F0F0F0Full);
    const T mask6 = static_cast<T>(0xF0F0F0F0F0F0F0F0ull);

    v = ((v & mask1) << 1) | ((v & mask2) >> 1);
    v = ((v & mask3) << 2) | ((v & mask4) >> 2);
    v = ((v & mask5) << 4) | ((v & mask6) >> 4);
    
    // Прямой возврат результата из веток исключает ошибки инициализации
    if consteval {
        return static_cast<T>(internal_byteswap(v) >> (sizeof(T) * 8 - bits));
    } else {
        return static_cast<T>(std::byteswap(v) >> (sizeof(T) * 8 - bits));
    }
}

// Тесты для проверки во время компиляции
static_assert(fast_bit_reverse(0b1101u, 4) == 0b1011u);
static_assert(fast_bit_reverse(0xACu, 8) == 0x35u);
static_assert(fast_bit_reverse(0x12345678u, 32) == 0x1E6A2C48u);

} // namespace utils
