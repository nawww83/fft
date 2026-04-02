#pragma once

#include <vector>
#include <complex>
#include <cstdint>
#include <memory>
#include <span>

/**
 * Краткие псевдонимы для целочисленных типов.
 * Соответствуют стандарту фиксированной ширины.
 */

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

// Дополнительно часто полезны типы для индексов
using f32 = float;
using f64 = double;

using Complex32 = std::complex<f32>; // TODO: реализовать общий БПФ.
using Complex64 = std::complex<f64>;

enum class FFTLayout {
    AoS, // [Re, Im, Re, Im] -> std::complex
    SoA  // [Re, Re...], [Im, Im...] -> 2 x double/float
};

enum class FFTType {
    Iterative,
    Recursive
};

// Обертки для типов данных
struct AoSData {
    std::span<Complex64> buffer;
};

struct SoAData {
    std::span<f64> re;
    std::span<f64> im;
};

#ifdef _MSC_VER
    #define RESTRICT __restrict
#else
    #define RESTRICT __restrict__
#endif

struct SwapPair 
{
    u32 i, j;
};

struct Swaps {
    explicit Swaps(size_t max_n);

    const std::vector<SwapPair>& get_for_n(size_t n) const;

    std::vector<std::vector<SwapPair>> rev_tables;
};

using SwapsSP = std::shared_ptr<const Swaps>;

struct TwiddleData {
    // Вектор для AoS (Complex64)
    std::vector<Complex64> aos;
    
    // Два вектора для SoA (раздельные Re и Im)
    std::vector<f64> soa_re;
    std::vector<f64> soa_im;
    
    // Оффсеты для начала каждой стадии (len = 8, 16, 32...)
    std::vector<size_t> table_offsets;

    explicit TwiddleData(size_t n);
};

using TwiddleSP = std::shared_ptr<const TwiddleData>;
