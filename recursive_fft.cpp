#include "recursive_fft.hpp"
#include "bit_reverse.hpp"

#include <numbers>

// --- 1. ВЫНОСИМ ГОРЯЧЕЕ ЯДРО СБОРКИ ДЛЯ SIMD ---
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
    __attribute__((target_clones("avx512f", "avx2", "avx", "default")))
#endif
static inline void apply_recursive_butterfly_simd(Complex* RESTRICT data, 
                                                 const Complex* RESTRICT twiddles, 
                                                 size_t half) 
{
    // Теперь это идеальный цикл для векторизации: два линейных чтения, два линейных записи
    for (size_t k = 0; k < half; ++k) {
        Complex w = twiddles[k]; 
        Complex u = data[k];
        Complex t = w * data[k + half];
        
        data[k]        = u + t;
        data[k + half] = u - t;
    }
}

FFTRecursive::FFTRecursive(size_t max_n) : m_max_n(max_n) {
    for (size_t n = 2; n <= max_n; n <<= 1) {
        ComplexVec direct, inverse;
        size_t half = n / 2;
        direct.reserve(half);
        inverse.reserve(half);
        
        for (size_t k = 0; k < half; ++k) {
            double ang = -2.0 * std::numbers::pi * k / n;
            direct.emplace_back(std::cos(ang), std::sin(ang));
            inverse.emplace_back(std::cos(-ang), std::sin(-ang));
        }
        m_twiddle_levels.push_back(std::move(direct));
        m_itwiddle_levels.push_back(std::move(inverse));
    }
}

void FFTRecursive::run_fft_inplace(Complex* data, size_t n, int level, 
                                   const std::vector<ComplexVec>& all_twiddles) 
{
    if (n <= 1) return;

    // Базовый случай: на малых N (32 и меньше) переходим на итеративный микроблок
    // или оставляем простую рекурсию, так как данные уже в L1 кэше.
    if (n <= 32) { 
        // Простая "бабочка" для малых длин без лишних вызовов
        for (size_t len = 2; len <= n; len <<= 1) {
            size_t half = len >> 1;
            const Complex* tw = all_twiddles[std::countr_zero(len)-1].data();
            for (size_t start = 0; start < n; start += len) {
                for (size_t j = 0; j < half; ++j) {
                    Complex t = tw[j] * data[start + j + half];
                    Complex u = data[start + j];
                    data[start + j] = u + t;
                    data[start + j + half] = u - t;
                }
            }
        }
        return;
    }

    size_t half = n / 2;
    // Рекурсивные вызовы (level-1)
    run_fft_inplace(data, half, level - 1, all_twiddles);
    run_fft_inplace(data + half, half, level - 1, all_twiddles);

    // Сборка текущего уровня (используем SIMD-ядро)
    // Коэффициенты берем из таблицы именно для этого уровня (level)
    apply_recursive_butterfly_simd(data, all_twiddles[level].data(), half);
}

void FFTRecursive::transform(ComplexVec& v, bool invert) {
    const size_t n = v.size();
    if (n <= 1) return;

    // 1. Bit-reversal (обязателен для In-place)
    const int log2n = std::countr_zero(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t j = utils::fast_bit_reverse(i, log2n);
        if (i < j) std::swap(v[i], v[j]);
    }

    // 2. Запуск рекурсии с правильной пирамидой таблиц
    run_fft_inplace(v.data(), n, log2n - 1, invert ? m_itwiddle_levels : m_twiddle_levels);

    if (invert) {
        double inv_n = 1.0 / n;
        for (auto& x : v) x *= inv_n;
    }
}
