#include "recursive_fft.hpp"
#include "bit_reverse.hpp"
#include <numbers>
#include <bit>
#include <algorithm>
#include <stdexcept>

// 1. КОНСТРУКТОР (Исправляем LNK2019)
FFTRecursive::FFTRecursive(size_t max_n) : m_max_n(max_n) {
    if (!std::has_single_bit(max_n)) {
        throw std::invalid_argument("FFT size must be a power of 2");
    }

    m_twiddles = generate_twiddles(max_n, false);
    m_itwiddles = generate_twiddles(max_n, true);
}

// --- 1. ВЫНОСИМ ГОРЯЧЕЕ ЯДРО СБОРКИ ДЛЯ SIMD ---
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
    __attribute__((target_clones("avx2", "avx", "default")))
#endif
static void apply_recursive_butterfly(Complex* RESTRICT data, 
                                     const Complex* RESTRICT twiddles, 
                                     size_t half, size_t stride) 
{
    // Этот цикл теперь будет векторизован компилятором под разные архитектуры
    for (size_t k = 0; k < half; ++k) {
        Complex w = twiddles[k * stride];
        Complex t = w * data[k + half];
        Complex u = data[k];
        
        data[k]        = u + t;
        data[k + half] = u - t;
    }
}

// --- 2. БАЗОВЫЙ ИТЕРАТИВНЫЙ БЛОК (ТОЖЕ С КЛОНАМИ) ---
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
    __attribute__((target_clones("avx2", "avx", "default")))
#endif
static void fast_base_fft(Complex* RESTRICT data, const Complex* RESTRICT twiddles, 
                         size_t n, size_t stride_base) 
{
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half = len >> 1;
        size_t step = stride_base * (n / len);
        for (size_t start = 0; start < n; start += len) {
            for (size_t j = 0; j < half; ++j) {
                Complex w = twiddles[j * step];
                Complex t = w * data[start + j + half];
                Complex u = data[start + j];
                data[start + j]        = u + t;
                data[start + j + half] = u - t;
            }
        }
    }
}

void FFTRecursive::run_fft_inplace(std::span<Complex> data, 
                                   const ComplexVec& twiddles, 
                                   size_t stride) 
{
    const size_t n = data.size();

    // Прерывание рекурсии (Threshold)
    if (n <= 16) {
        fast_base_fft(data.data(), twiddles.data(), n, stride);
        return;
    }

    const size_t half = n / 2;

    // Рекурсия
    run_fft_inplace(data.subspan(0, half), twiddles, stride * 2);
    run_fft_inplace(data.subspan(half, half), twiddles, stride * 2);

    // Вызов мультиверсионного ядра сборки
    apply_recursive_butterfly(data.data(), twiddles.data(), half, stride);
}

// 3. ОСНОВНОЙ МЕТОД
void FFTRecursive::transform(ComplexVec& v, bool invert) {
    const size_t n = v.size();
    if (n <= 1) return;
    
    if (n > m_max_n || !std::has_single_bit(n)) {
        throw std::invalid_argument("Invalid FFT size");
    }

    // Шаг 1: Bit-reversal перестановка (теперь обязательна для In-place рекурсии)
    const int log2n = static_cast<int>(std::countr_zero(n));
    for (size_t i = 0; i < n; ++i) {
        size_t j = utils::fast_bit_reverse(i, log2n);
        if (i < j) std::swap(v[i], v[j]);
    }

    // Шаг 2: Рекурсивная сборка
    const ComplexVec& tw = invert ? m_itwiddles : m_twiddles;
    run_fft_inplace(std::span<Complex>(v), tw, m_max_n / n);

    // Шаг 3: Нормализация
    if (invert) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto& x : v) x *= inv_n;
    }
}

// Генерируем LUT
ComplexVec FFTRecursive::generate_twiddles(size_t n, bool invert) {
    ComplexVec res;
    res.reserve(n / 2);
    const double angle_base = (invert ? 2.0 : -2.0) * std::numbers::pi / n;
    for (size_t k = 0; k < n / 2; ++k) {
        res.emplace_back(std::polar(1.0, k * angle_base));
    }
    return res;
}
