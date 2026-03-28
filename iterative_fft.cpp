#include "iterative_fft.hpp"
#include <numbers>
#include <bit>
#include <algorithm>
#include <complex>

#include <ranges>   // C++20/23
#include <numeric>

#include "iterative_fft.hpp"

// 1. ВЫНОСИМ ЯДРО В ОТДЕЛЬНУЮ ФУНКЦИЮ ДЛЯ SIMD-КЛОНИРОВАНИЯ
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
    __attribute__((target_clones("avx2", "avx", "sse4.2", "default")))
#endif
static void apply_butterfly_block(Complex* RESTRICT data_low, 
                                 Complex* RESTRICT data_high, 
                                 const Complex* RESTRICT table, 
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

// Базовая бабочка без умножения (для N=2)
inline void butterfly2(Complex& a, Complex& b) {
    Complex t = b;
    b = a - t;
    a = a + t;
}

// Базовая бабочка с умножением на -i (для N=4)
inline void butterfly4_is(Complex& a, Complex& b) {
    // b * (-i) = {b.imag, -b.real}
    Complex t = {b.imag(), -b.real()};
    b = a - t;
    a = a + t;
}

void FFTIterative::transform(ComplexVec &v, bool invert) const {
    const size_t n = v.size();
    if (n <= 1) return;

    // 1. Bit-reversal (без изменений)
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(v[i], v[j]);
    }

    // 2. БАЗОВЫЕ СЛУЧАИ (Развернутые слои)
    
    // Слой len=2: w всегда 1
    for (size_t i = 0; i < n; i += 2) {
        butterfly2(v[i], v[i + 1]);
    }

    // Слой len=4: w это 1 и -i (или i)
    if (n >= 4) {
        for (size_t i = 0; i < n; i += 4) {
            // j = 0: w = 1
            butterfly2(v[i], v[i + 2]);
            // j = 1: w = -i (forward) или i (inverse)
            if (!invert) butterfly4_is(v[i + 1], v[i + 3]);
            else {
                Complex t = {-v[i + 3].imag(), v[i + 3].real()}; // * i
                v[i + 3] = v[i + 1] - t;
                v[i + 1] = v[i + 1] + t;
            }
        }
    }

    // 3. ОСНОВНОЙ ЦИКЛ (начиная с len=8)
    for (size_t len = 8; len <= n; len <<= 1) {
        const size_t half = len >> 1;
        const size_t step = max_n / len;

        for (size_t start = 0; start < n; start += len) {
            apply_butterfly_block(
                v.data() + start, 
                v.data() + start + half, 
                table.data(), 
                half, 
                step, 
                invert
            );
        }
    }

    if (invert) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto &x : v) x *= inv_n;
    }
}
