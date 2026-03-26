#include "recursive_fft.hpp"
#include <numbers>
#include <bit>
#include <ranges>
#include <algorithm>

// Реализация SIMD-ядра
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target_clones("avx2", "avx", "sse4.2", "default")))
#endif
static void compute_butterfly_simd(Complex* __restrict__ out,
                                   const Complex* __restrict__ res_e,
                                   const Complex* __restrict__ res_o,
                                   const Complex* __restrict__ tw,
                                   size_t half_n,
                                   size_t stride) 
{
    for (size_t k = 0; k < half_n; ++k) {
        Complex t = tw[k * stride] * res_o[k];
        out[k]          = res_e[k] + t;
        out[k + half_n] = res_e[k] - t;
    }
}

FFTRecursive::FFTRecursive(size_t max_n) : m_max_n(max_n) {
    if (!std::has_single_bit(max_n))
        throw std::invalid_argument("Размер БПФ должен быть степенью двойки");

    m_buffer.resize(max_n);
    m_twiddles = generate_twiddles(max_n, false);
    m_itwiddles = generate_twiddles(max_n, true);
}

ComplexVec FFTRecursive::generate_twiddles(size_t n, bool invert) {
    const double angle_base = (invert ? 2.0 : -2.0) * std::numbers::pi / n;
    
    // 1. Создаем ленивое представление
    auto view = std::views::iota(0u, n / 2) 
              | std::views::transform([=](size_t k) { 
                    return std::polar(1.0, k * angle_base); 
                });

    // 2. Явно копируем в вектор (это сработает везде)
    ComplexVec res;
    res.reserve(n / 2);
    for (auto val : view) {
        res.push_back(val);
    }
    return res;
}

void FFTRecursive::transform(ComplexVec& v, bool invert) {
    const size_t n = v.size();
    if (n <= 1) return;
    if (n > m_max_n || !std::has_single_bit(n))
        throw std::invalid_argument("Недопустимый размер входного вектора");

    // Выбираем нужную таблицу и считаем начальный шаг (stride)
    const ComplexVec& tw = invert ? m_itwiddles : m_twiddles;
    size_t initial_stride = m_max_n / n;

    run_fft(v, std::span(m_buffer).subspan(0, n), tw, initial_stride);

    if (invert) {
        double inv_n = 1.0 / static_cast<double>(n);
        for (auto& x : v) x *= inv_n;
    }
}

void FFTRecursive::run_fft(std::span<Complex> data,
                           std::span<Complex> buffer,
                           const ComplexVec& twiddles,
                           size_t current_stride) 
{
    const size_t n = data.size();
    if (n <= 1) return;
    const size_t half = n / 2;

    // Распределяем элементы по четности (copy-stride)
    for (size_t i = 0; i < half; ++i) {
        buffer[i]        = data[i * 2];
        buffer[i + half] = data[i * 2 + 1];
    }

    // Рекурсия: меняем роли data и buffer для экономии памяти
    run_fft(buffer.subspan(0, half), data.subspan(0, half), twiddles, current_stride * 2);
    run_fft(buffer.subspan(half, half), data.subspan(half, half), twiddles, current_stride * 2);

    // Бабочка с использованием SIMD-ядра
    // Передаем указатели и размер напрямую
    compute_butterfly_simd(
        data.data(), 
        buffer.data(), 
        buffer.data() + half, 
        twiddles.data(), 
        half, 
        current_stride
    );
}
