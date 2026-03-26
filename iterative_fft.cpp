#include "iterative_fft.hpp"
#include <numbers>
#include <bit>
#include <algorithm>
#include <complex>

// SIMD-ядро для обработки одного блока (Butterfly layer)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target_clones("avx2", "avx", "sse4.2", "default")))
#endif
static void apply_butterfly_block(Complex* __restrict__ data_low, 
                                 Complex* __restrict__ data_high, 
                                 const Complex* __restrict__ table, 
                                 size_t half, size_t step, bool invert) 
{
    // Внутренний цикл теперь идет по j (0...half)
    // Данные читаются из памяти последовательно, что идеально для SIMD
    for (size_t j = 0; j < half; ++j) {
        Complex w = table[j * step];
        if (invert) w = std::conj(w);

        Complex u = data_low[j];
        Complex t = data_high[j] * w;
        
        data_low[j]  = u + t;
        data_high[j] = u - t;
    }
}

FFTIterative::FFTIterative(size_t max_n) : max_n(max_n) {
    table.reserve(max_n / 2);
    const double angle_base = -2.0 * std::numbers::pi / max_n;
    for (size_t i = 0; i < max_n / 2; ++i) {
        table.emplace_back(std::polar(1.0, i * angle_base));
    }
}

void FFTIterative::transform(ComplexVec &v, bool invert) const {
    const size_t n = v.size();
    if (n <= 1) return;

    // 1. Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(v[i], v[j]);
    }

    // 2. Итеративные проходы (Layers)
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t half = len >> 1;
        const size_t step = max_n / len;

        // Внешний цикл идет по блокам данных размера len
        for (size_t start = 0; start < n; start += len) {
            // Вызываем SIMD-ядро для конкретного блока
            apply_butterfly_block(
                v.data() + start,        // "Нижняя" половина бабочек
                v.data() + start + half, // "Верхняя" половина
                table.data(), 
                half, 
                step, 
                invert
            );
        }
    }

    // 3. Нормировка для обратного преобразования
    if (invert) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto &x : v) x *= inv_n;
    }
}
