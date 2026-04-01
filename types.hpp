#pragma once

#include <vector>
#include <complex>
#include <cstdint>

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

using RealVec = std::vector<f64>;
using Complex = std::complex<f64>;
using ComplexVec = std::vector<Complex>;

// Исправляем ключевое слово restrict для MSVC
#ifdef _MSC_VER
    #define RESTRICT __restrict
#else
    #define RESTRICT __restrict__
#endif