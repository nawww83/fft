#include "types.hpp"
#include <bit>
#include <concepts>
#include <cassert>
#include <numbers>

namespace {

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

}

Swaps::Swaps(size_t max_n)
{
    if (max_n < 2) return;
    
    size_t max_log2 = std::bit_width(max_n) - 1;
    rev_tables.resize(max_log2 + 1);

    for (size_t n = 2; n <= max_n; n <<= 1) {
        int log2n = std::bit_width(n) - 1;
        std::vector<SwapPair> pairs;
        
        for (u32 i = 0; i < (u32)n; ++i) {
            u32 j = fast_bit_reverse(i, log2n);
            if (i < j) pairs.push_back({i, j});
        }
        rev_tables[log2n] = std::move(pairs);
    }
}

const std::vector<SwapPair> &Swaps::get_for_n(size_t n) const
{
    size_t idx = std::countr_zero(n);
    if (idx >= rev_tables.size()) { // Убираем проверку на empty()
        throw std::out_of_range("Swaps table not precomputed for this N");
    }
    return rev_tables[idx];
}

TwiddleData::TwiddleData(size_t n)
{
    if (n < 8) return;

    // Точный расчет: для n=1024 сумма (4+8+16...+512) = 1020
    // Резервируем с запасом n, этого точно хватит
    aos.reserve(n);
    soa_re.reserve(n);
    soa_im.reserve(n);
    
    // Оффсетов будет log2(n) - 2 (так как стартуем с 8, а не с 2)
    table_offsets.reserve(32); 

    for (size_t len = 8; len <= n; len <<= 1) {
        table_offsets.push_back(soa_re.size()); // Используем текущий размер как оффсет
        
        const size_t half = len >> 1;
        const f64 angle_step = -2.0 * std::numbers::pi / static_cast<f64>(len);

        for (size_t j = 0; j < half; ++j) {
            f64 angle = angle_step * j;
            f64 c = std::cos(angle);
            f64 s = std::sin(angle);

            aos.emplace_back(c, s);
            soa_re.push_back(c);
            soa_im.push_back(s);
        }
    }
}
