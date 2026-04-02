#pragma once

#include "base_fft.hpp"

#include <numbers>
#include <utility>
#include <bit>

/**
 * @brief Базовая бабочка Radix-2 (W = 1)
 */
inline void butterfly2(Complex64 &a, Complex64 &b)
{
    Complex64 t = b;
    b = a - t;
    a = a + t;
}

/**
 * @brief Оптимизированная бабочка для W = -i (Imaginary Shift)
 * Используется в слое N=4 для избежания честного комплексного умножения.
 */
inline void butterfly4_is(Complex64 &a, Complex64 &b)
{
    // b * (-i) <=> {b.imag, -b.real}
    Complex64 t(b.imag(), -b.real());
    b = a - t;
    a = a + t;
}

struct Direct
{
    static inline Complex64 get(Complex64 w) { return w; }
};

struct Inverse
{
    static inline Complex64 get(Complex64 w) { return std::conj(w); }
};

/**
 * @brief Ядро бабочки FFT, вынесенное для SIMD векторизации.
 * target_clones позволяет компилятору создать версии под разные наборы инструкций.
 */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
__attribute__((target_clones("avx512f", "avx2", "default")))
#endif
template <typename Mode>
static inline void apply_butterfly_block_simd(Complex64 *__restrict data_low,
                                              Complex64 *__restrict data_high,
                                              const Complex64 *__restrict table,
                                              size_t half)
{
    for (size_t j = 0; j < half; ++j)
    {
        Complex64 w = Mode::get(table[j]);
        Complex64 u = data_low[j];
        Complex64 t = data_high[j] * w;

        data_low[j] = u + t;
        data_high[j] = u - t;
    }
}

class FFTIterativeAoS : public FFTBase
{
public:
    using FFTBase::FFTBase;

    FFTLayout getLayout() const override { return FFTLayout::AoS; }

    void transform(AoSData &data, bool invert) override
    {
        const auto n = data.buffer.size();
        if (n <= 1)
            return;
        const auto &pairs = m_swaps->get_for_n(n);

        // Bit-reversal swap
        for (const auto &p : pairs)
        {
            std::swap(data.buffer[p.i], data.buffer[p.j]);
        }

        // Вычисления БПФ (AoS-style)
        if (invert)
        {
            execute_aos<true>(data.buffer);
        }
        else
        {
            execute_aos<false>(data.buffer);
        }
    }

private:
    template <bool invert>
    void execute_aos(std::span<Complex64> v) const
    {
        const size_t n = v.size();

        // Слой len=2
        for (size_t i = 0; i < n; i += 2)
        {
            butterfly2(v[i], v[i + 1]);
        }

        // Слой len=4
        if (n >= 4)
        {
            for (size_t i = 0; i < n; i += 4)
            {
                butterfly2(v[i], v[i + 2]);
                if constexpr (!invert)
                {
                    butterfly4_is(v[i + 1], v[i + 3]);
                }
                else
                {
                    Complex64 t(-v[i + 3].imag(), v[i + 3].real());
                    v[i + 3] = v[i + 1] - t;
                    v[i + 1] = v[i + 1] + t;
                }
            }
        }

        // Основной цикл
        size_t table_idx = 0;
        for (size_t len = 8; len <= n; len <<= 1)
        {
            const size_t half = len >> 1;
            const Complex64 *current_table = &m_twiddles->aos[m_twiddles->table_offsets[table_idx++]];
            for (size_t start = 0; start < n; start += len)
            {
                if constexpr (invert)
                {
                    apply_butterfly_block_simd<Inverse>(v.data() + start,
                                                        v.data() + start + half,
                                                        current_table,
                                                        half);
                }
                else
                {
                    apply_butterfly_block_simd<Direct>(v.data() + start,
                                                       v.data() + start + half,
                                                       current_table,
                                                       half);
                }
            }
        }

        if constexpr (invert)
        {
            const f64 inv_n = 1.0 / n;
            for (auto &x : v)
                x *= inv_n;
        }
    }
};
