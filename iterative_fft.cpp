#include "iterative_fft.hpp"
#include "bit_reverse.hpp"
#include <numbers>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <bit>
#include <cmath>

/**
 * @brief Ядро бабочки FFT, вынесенное для SIMD векторизации.
 * target_clones позволяет компилятору создать версии под разные наборы инструкций.
 */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
__attribute__((target_clones("avx2", "avx", "default")))
#endif
static void apply_butterfly_block(Complex *RESTRICT data_low,
                                  Complex *RESTRICT data_high,
                                  const Complex *RESTRICT table,
                                  size_t half, bool invert)
{
    // Внутренний цикл по j обеспечивает линейный проход по LUT (step=1)
    for (size_t j = 0; j < half; ++j)
    {
        Complex w = table[j];
        if (invert)
            w = std::conj(w);

        Complex u = data_low[j];
        Complex t = data_high[j] * w;

        data_low[j] = u + t;
        data_high[j] = u - t;
    }
}

FFTIterative::FFTIterative(size_t max_n) : max_n(max_n)
{
    // Предварительный расчет таблиц поворотных коэффициентов (LUT)
    // Суммарный объем: N/4 + N/8 + ... + 2 + 1 ≈ N/2 элементов.
    full_table.reserve(max_n / 2);

    // Генерируем LUT начиная с len=8, так как уровни 2 и 4 оптимизированы вручную
    for (size_t len = 8; len <= max_n; len <<= 1)
    {
        table_offsets.push_back(full_table.size());

        const size_t half = len >> 1;
        const double angle_base = -2.0 * std::numbers::pi / len;

        for (size_t j = 0; j < half; ++j)
        {
            full_table.emplace_back(std::polar(1.0, j * angle_base));
        }
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

void FFTIterative::transform(ComplexVec &v, bool invert) const
{
    const size_t n = v.size();
    if (n <= 1)
        return;

    // 1. Bit-reversal перестановка
    const int log2n = static_cast<int>(std::countr_zero(n));
    for (size_t i = 0; i < n; ++i)
    {
        size_t j = utils::fast_bit_reverse(i, log2n);
        if (i < j)
            std::swap(v[i], v[j]);
    }

    // 2. Базовые слои (Развернутые циклы для исключения LUT)

    // Слой len=2: поворотный коэффициент всегда 1
    for (size_t i = 0; i < n; i += 2)
    {
        butterfly2(v[i], v[i + 1]);
    }

    // Слой len=4: коэффициенты 1 и -i (или i при инверсии)
    if (n >= 4)
    {
        for (size_t i = 0; i < n; i += 4)
        {
            // j = 0: w = 1
            butterfly2(v[i], v[i + 2]);
            // j = 1: w = -i (forward) или w = i (inverse)
            if (!invert)
            {
                butterfly4_is(v[i + 1], v[i + 3]);
            }
            else
            {
                Complex t(-v[i + 3].imag(), v[i + 3].real()); // умножение на i
                v[i + 3] = v[i + 1] - t;
                v[i + 1] = v[i + 1] + t;
            }
        }
    }

    // 3. Основной цикл (от len=8 и выше)
    size_t table_idx = 0;
    for (size_t len = 8; len <= n; len <<= 1)
    {
        const size_t half = len >> 1;
        const Complex *current_table = &full_table[table_offsets[table_idx++]];

        for (size_t start = 0; start < n; start += len)
        {
            apply_butterfly_block(
                v.data() + start,
                v.data() + start + half,
                current_table,
                half,
                invert);
        }
    }

    // Нормализация при обратном преобразовании
    if (invert)
    {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto &x : v)
            x *= inv_n;
    }
}
