#include "recursive_fft_aos.hpp"
#include <numbers>
#include <bit>

// --- 1. ВЫНОСИМ ГОРЯЧЕЕ ЯДРО СБОРКИ ДЛЯ SIMD ---
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
__attribute__((target_clones("avx512f", "avx2", "default")))
#endif
static inline void apply_recursive_butterfly_simd(Complex64 *RESTRICT data,
                                                  const Complex64 *RESTRICT twiddles,
                                                  size_t half)
{
    // Теперь это идеальный цикл для векторизации: два линейных чтения, два линейных записи
    for (size_t k = 0; k < half; ++k)
    {
        Complex64 w = twiddles[k];
        Complex64 u = data[k];
        Complex64 t = w * data[k + half];

        data[k] = u + t;
        data[k + half] = u - t;
    }
}

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
__attribute__((target_clones("avx512f", "avx2", "default")))
#endif
static inline void apply_recursive_butterfly_inv_simd(Complex64 *RESTRICT data,
                                                      const Complex64 *RESTRICT twiddles,
                                                      size_t half)
{
    for (size_t k = 0; k < half; ++k)
    {
        // Конъюгация "на лету": (tw.re, -tw.im)
        Complex64 w = {twiddles[k].real(), -twiddles[k].imag()};
        Complex64 u = data[k];
        Complex64 t = w * data[k + half];

        data[k] = u + t;
        data[k + half] = u - t;
    }
}

void FFTRecursiveAoS::run_fft_inplace_fwd(Complex64 *data, size_t n,
                                          const TwiddleData &twiddles)
{
    // Базовый случай (Hybrid)
    if (n <= 32)
    {
        // 1. Слои len=2 и len=4 (без таблиц)
        for (size_t i = 0; i < n; i += 2)
        {
            Complex64 u = data[i], t = data[i + 1];
            data[i] = u + t;
            data[i + 1] = u - t;
        }
        if (n <= 2) return;

        for (size_t i = 0; i < n; i += 4)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                Complex64 w = (j == 0) ? Complex64{1, 0} : Complex64{0, -1};
                Complex64 t = w * data[i + j + 2];
                Complex64 u = data[i + j];
                data[i + j] = u + t;
                data[i + j + 2] = u - t;
            }
        }
        if (n <= 4) return;
    
        // 2. Слои len=8...32 (из таблиц TwiddleData)
        size_t t_idx = 0;
        for (size_t len = 8; len <= n; len <<= 1)
        {
            size_t half = len >> 1;
            const Complex64 *tw = &twiddles.aos[twiddles.table_offsets[t_idx++]];
            for (size_t start = 0; start < n; start += len)
            {
                apply_recursive_butterfly_simd(data + start, tw, half);
            }
        }
        return;
    }

    size_t half = n / 2;
    run_fft_inplace_fwd(data, half, twiddles);
    run_fft_inplace_fwd(data + half, half, twiddles);

    // Сборка уровня: находим индекс в таблице для текущего n
    // Если n=64, это 4-й слой после 2,4,8,16,32.
    // В TwiddleData слои начинаются с 8, значит для n=64 t_idx = log2(64)-3 = 3.
    size_t t_idx = std::bit_width(n) - 1 - 3;
    apply_recursive_butterfly_simd(data, &twiddles.aos[twiddles.table_offsets[t_idx]], half);
}

void FFTRecursiveAoS::run_fft_inplace_inv(Complex64 *data, size_t n, const TwiddleData &twiddles)
{
    // --- БАЗОВЫЙ СЛУЧАЙ (Inverse Hybrid N <= 32) ---
    if (n <= 32)
    {
        // 1. Слой len=2 (такой же как в fwd)
        for (size_t i = 0; i < n; i += 2)
        {
            Complex64 u = data[i], t = data[i + 1];
            data[i] = u + t;
            data[i + 1] = u - t;
        }
        if (n <= 2) return;

        // 2. Слой len=4 (w = +i для inverse)
        for (size_t i = 0; i < n; i += 4)
        {
            // j=0 (w=1)
            Complex64 u0 = data[i], t0 = data[i + 2];
            data[i] = u0 + t0;
            data[i + 2] = u0 - t0;
            // j=1 (w=+i для inverse)
            Complex64 u1 = data[i + 1], v1 = data[i + 3];
            Complex64 t1 = {-v1.imag(), v1.real()}; // v * (0 + i)
            data[i + 1] = u1 + t1;
            data[i + 3] = u1 - t1;
        }
        if (n <= 4) return;

        // 3. Слои len=8...32 (из таблиц с инверсией Imag)
        size_t t_idx = 0;
        for (size_t len = 8; len <= n; len <<= 1)
        {
            size_t half = len >> 1;
            const Complex64 *tw = &twiddles.aos[twiddles.table_offsets[t_idx++]];
            for (size_t start = 0; start < n; start += len)
            {
                // Используем специальное ядро для инверсии
                apply_recursive_butterfly_inv_simd(data + start, tw, half);
            }
        }
        return;
    }

    // --- РЕКУРСИВНЫЙ ШАГ ---
    size_t half = n / 2;
    run_fft_inplace_inv(data, half, twiddles);
    run_fft_inplace_inv(data + half, half, twiddles);

    // --- СБОРКА УРОВНЯ ---
    size_t t_idx = std::bit_width(n) - 1 - 3;
    const Complex64 *tw = &twiddles.aos[twiddles.table_offsets[t_idx]];

    apply_recursive_butterfly_inv_simd(data, tw, half);
}

void FFTRecursiveAoS::transform(AoSData &data, bool invert)
{
    const size_t n = data.buffer.size();
    if (n <= 1)
        return;

    // 1. Bit-reversal (обязателен для In-place DIT рекурсии)
    const auto &pairs = m_swaps->get_for_n(n);
    Complex64 *ptr = data.buffer.data();
    for (const auto [i, j] : pairs)
    {
        std::swap(ptr[i], ptr[j]);
    }

    // 2. Запуск рекурсии
    // Передаем m_twiddles по ссылке для скорости
    if (invert)
    {
        run_fft_inplace_inv(ptr, n, *m_twiddles);
    }
    else
    {
        run_fft_inplace_fwd(ptr, n, *m_twiddles);
    }

    // 3. Нормировка при обратном преобразовании
    if (invert)
    {
        f64 inv_n = 1.0 / static_cast<f64>(n);
        for (size_t i = 0; i < n; ++i)
        {
            ptr[i] *= inv_n;
        }
    }
}
