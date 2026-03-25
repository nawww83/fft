#include <numbers>
#include <bit>
#include <algorithm>
#include <limits>
#include <cmath>

#include <ranges>   // C++20/23
#include <numeric>

#include "iterative_fft.hpp"

// 1. ВЫНОСИМ ЯДРО В ОТДЕЛЬНУЮ ФУНКЦИЮ ДЛЯ SIMD-КЛОНИРОВАНИЯ
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target_clones("avx2", "avx", "sse4.2", "default")))
#endif
static void apply_butterfly_layer(Complex* __restrict__ data, 
                           const Complex* __restrict__ table, 
                           size_t n, size_t len, size_t half, size_t step, 
                           bool invert) 
{
    // Теперь компилятор создаст несколько версий этой функции под разные наборы инструкций
    for (size_t j = 0; j < half; j++) {
        Complex w = table[j * step];
        if (invert) w = std::conj(w);

        // Этот цикл станет векторизованным (SIMD)
        for (size_t i = j; i < n; i += len) {
            Complex u = data[i];
            Complex t = data[i + half] * w;
            data[i] = u + t;
            data[i + half] = u - t;
        }
    }
}

FFTIterative::FFTIterative(size_t max_n) : max_n(max_n)
{
    table.reserve(max_n / 2);
    double angle_base = -2.0 * std::numbers::pi / max_n;
    for (size_t i = 0; i < max_n / 2; ++i)
    {
        table.emplace_back(std::polar(1.0, i * angle_base));
    }
}

void FFTIterative::transform(std::vector<Complex> &v, bool invert) const
{
    size_t n = v.size();
    if (n <= 1)
        return;

    // Bit-reversal (оставляем как есть)
    for (size_t i = 1, j = 0; i < n; i++)
    {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(v[i], v[j]);
    }

    // 2. ВЫЗОВ ОПТИМИЗИРОВАННОГО ЯДРА
    for (size_t len = 2; len <= n; len <<= 1)
    {
        size_t half = len >> 1;
        size_t step = max_n / len;

        // Передаем сырые указатели, чтобы компилятору было проще с SIMD
        apply_butterfly_layer(v.data(), table.data(), n, len, half, step, invert);
    }

    if (invert)
    {
        double inv_n = 1.0 / n;
        for (auto &x : v)
            x *= inv_n;
    }
}
