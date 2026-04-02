#include "recursive_fft_soa.hpp"
#include <bit>

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
__attribute__((target_clones("avx512f", "avx2", "default")))
#endif
static inline void apply_butterfly_soa_simd(f64 *RESTRICT re, f64 *RESTRICT im,
                                            const f64 *RESTRICT tw_r,
                                            const f64 *RESTRICT tw_i,
                                            size_t half, bool invert)
{
    const f64 s = invert ? -1 : 1;
    for (size_t k = 0; k < half; ++k)
    {
        f64 wr = tw_r[k];
        f64 wi = s * tw_i[k]; // Сопряжение для INV

        f64 dr = re[k + half], di = im[k + half];

        f64 tr = wr * dr - wi * di;
        f64 ti = wr * di + wi * dr;

        f64 ur = re[k], ui = im[k];

        re[k] = ur + tr;
        im[k] = ui + ti;
        re[k + half] = ur - tr;
        im[k + half] = ui - ti;
    }
}

void FFTRecursiveSoA::transform(SoAData &data, bool invert)
{
    const size_t n = data.re.size();
    if (n <= 1)
        return;

    f64 *RESTRICT pr = data.re.data();
    f64 *RESTRICT pi = data.im.data();

    // 1. Bit-reversal
    const auto &pairs = m_swaps->get_for_n(n);
    for (const auto [i, j] : pairs)
    {
        std::swap(pr[i], pr[j]);
        std::swap(pi[i], pi[j]);
    }

    // 2. Рекурсия
    run_fft_soa(pr, pi, n, *m_twiddles, invert);

    // 3. Нормировка
    if (invert)
    {
        f64 inv_n = 1.0 / n;
        for (size_t k = 0; k < n; ++k)
        {
            pr[k] *= inv_n;
            pi[k] *= inv_n;
        }
    }
}

void FFTRecursiveSoA::run_fft_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t n, const TwiddleData &tw, bool invert)
{
    if (n <= 32) {
        // 1. Слой 2 (Аналог AoS)
        for (size_t i = 0; i < n; i += 2) {
            f64 ur = re[i], ui = im[i];
            f64 tr = re[i+1], ti = im[i+1];
            re[i] = ur + tr; im[i] = ui + ti;
            re[i+1] = ur - tr; im[i+1] = ui - ti;
        }
        if (n <= 2) return;

        // 2. Слой 4
        // Forward: w = (1,0) и (0,-1)
        // Inverse: w = (1,0) и (0, 1)
        f64 w1_im = invert ? 1.0 : -1.0; 
        for (size_t i = 0; i < n; i += 4) {
            // j = 0 (w = 1 + 0i)
            f64 ur0 = re[i], ui0 = im[i];
            f64 tr0 = re[i+2], ti0 = im[i+2];
            re[i] = ur0 + tr0;   im[i] = ui0 + ti0;
            re[i+2] = ur0 - tr0; im[i+2] = ui0 - ti0;

            // j = 1 (w = 0 + i*w1_im)
            f64 ur1 = re[i+1], ui1 = im[i+1];
            // Комплексное умножение: (0 + i*w1_im) * (re[i+3] + i*im[i+3])
            // tr = -w1_im * im[i+3], ti = w1_im * re[i+3]
            f64 tr1 = -w1_im * im[i+3];
            f64 ti1 =  w1_im * re[i+3];
            re[i+1] = ur1 + tr1; im[i+1] = ui1 + ti1;
            re[i+3] = ur1 - tr1; im[i+3] = ui1 - ti1;
        }
        if (n <= 4) return;

        // 3. Слои 8..32 (Итеративная часть гибрида)
        size_t t_local_idx = 0; // Локальный индекс для слоев внутри базового случая
        for (size_t len = 8; len <= n; len <<= 1) {
            size_t half = len >> 1;
            for (size_t start = 0; start < n; start += len) {
                apply_butterfly_soa_simd(re + start, im + start,
                                         &tw.soa_re[tw.table_offsets[t_local_idx]],
                                         &tw.soa_im[tw.table_offsets[t_local_idx]],
                                         half, invert);
            }
            t_local_idx++;
        }
        return;
    }

    // РЕКУРСИЯ
    size_t half = n / 2;
    run_fft_soa(re, im, half, tw, invert);
    run_fft_soa(re + half, im + half, half, tw, invert);

    // Сборка (t_idx должен соответствовать n)
    size_t t_idx = std::bit_width(n) - 1 - 3; 
    apply_butterfly_soa_simd(re, im, &tw.soa_re[tw.table_offsets[t_idx]],
                             &tw.soa_im[tw.table_offsets[t_idx]], half, invert);
}
