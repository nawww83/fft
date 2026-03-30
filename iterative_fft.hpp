#pragma once

#include "types.hpp"
#include "bit_reverse.hpp"

#include <numbers>
#include <utility>
#include <bit>

/**
 * @brief Ядро бабочки FFT, вынесенное для SIMD векторизации.
 * target_clones позволяет компилятору создать версии под разные наборы инструкций.
 */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
__attribute__((target_clones("avx2", "avx", "default")))
#endif
template <bool invert>
static inline void apply_butterfly_block(Complex *__restrict data_low,
                                  Complex *__restrict data_high,
                                  const Complex *__restrict table,
                                  size_t half)
{
    for (size_t j = 0; j < half; ++j)
    {
        // Компилятор выберет одну ветку и полностью удалит вторую
        Complex w = table[j];
        if constexpr (invert) {
            w = std::conj(w);
        }

        Complex u = data_low[j];
        Complex t = data_high[j] * w;

        data_low[j] = u + t;
        data_high[j] = u - t;
    }
}

/**
 * @brief Базовая бабочка Radix-2 (W = 1)
 */
inline void butterfly2(Complex &a, Complex &b)
{
    Complex t = b;
    b = a - t;
    a = a + t;
}

/**
 * @brief Оптимизированная бабочка для W = -i (Imaginary Shift)
 * Используется в слое N=4 для избежания честного комплексного умножения.
 */
inline void butterfly4_is(Complex &a, Complex &b)
{
    // b * (-i) <=> {b.imag, -b.real}
    Complex t(b.imag(), -b.real());
    b = a - t;
    a = a + t;
}

class FFTIterative {
public:
    explicit FFTIterative(size_t max_n);

    void transform(ComplexVec& v, bool invert = false) const;

private:
    size_t max_n;
    ComplexVec full_table;
    std::vector<size_t> table_offsets; // Указатели на начало таблиц для каждого len

template <bool invert>
void execute_transform(ComplexVec &v) const {
    const size_t n = v.size();
    const int log2n = static_cast<int>(std::countr_zero(n));

    // 1. Bit-reversal (остается как было)
    for (size_t i = 0; i < n; ++i) {
        size_t j = utils::fast_bit_reverse(i, log2n);
        if (i < j) std::swap(v[i], v[j]);
    }

    // 2. Слой len=2
    for (size_t i = 0; i < n; i += 2) {
        butterfly2(v[i], v[i + 1]);
    }

    // 3. Слой len=4
    if (n >= 4) {
        for (size_t i = 0; i < n; i += 4) {
            butterfly2(v[i], v[i + 2]);
            if constexpr (!invert) {
                butterfly4_is(v[i + 1], v[i + 3]);
            } else {
                // Оптимизированный сопряженный вариант для invert=true
                Complex t(-v[i + 3].imag(), v[i + 3].real()); 
                v[i + 3] = v[i + 1] - t;
                v[i + 1] = v[i + 1] + t;
            }
        }
    }

    // 4. Основной цикл
    size_t table_idx = 0;
    for (size_t len = 8; len <= n; len <<= 1) {
        const size_t half = len >> 1;
        const Complex *current_table = &full_table[table_offsets[table_idx++]];
        for (size_t start = 0; start < n; start += len) {
            apply_butterfly_block<invert>( // Передаем параметр шаблона дальше
                v.data() + start,
                v.data() + start + half,
                current_table,
                half);
        }
    }

    if constexpr (invert) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto &x : v) x *= inv_n;
    }
}

};
