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

    // 1. Bit-reversal (дважды)
    const auto &pairs = m_swaps->get_for_n(n);
    for (const auto &p : pairs)
    {
        std::swap(pr[p.i], pr[p.j]);
        std::swap(pi[p.i], pi[p.j]);
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
        // Слой 2
        for (size_t i = 0; i < n; i += 2) {
            f64 ur = re[i], tr = re[i+1], ui = im[i], ti = im[i+1];
            re[i] = ur + tr; re[i+1] = ur - tr;
            im[i] = ui + ti; im[i+1] = ui - ti;
        }
        if (n <= 2) return; // Защита для малых N

        // Слой 4
        for (size_t i = 0; i < n; i += 4) {
            for (size_t j = 0; j < 2; ++j) {
                f64 wr = (j == 0) ? 1.0 : 0.0;
                f64 wi = (j == 0) ? 0.0 : (invert ? 1.0 : -1.0);
                f64 tr = wr * re[i+j+2] - wi * im[i+j+2];
                f64 ti = wr * im[i+j+2] + wi * re[i+j+2];
                f64 ur = re[i+j], ui = im[i+j];
                re[i+j] = ur + tr; re[i+j+2] = ur - tr;
                im[i+j] = ui + ti; im[i+j+2] = ui - ti;
            }
        }
        if (n <= 4) return; // Защита

        // Слои 8..32 — только если таблицы вообще существуют!
        if (tw.table_offsets.empty()) return;

        for (size_t len = 8; len <= n; len <<= 1) {
            size_t layer_idx = std::bit_width(len) - 1 - 3;
            // Проверка, что фабрика была создана с достаточным max_n
            if (layer_idx >= tw.table_offsets.size()) break;

            size_t half = len >> 1;
            for (size_t start = 0; start < n; start += len) {
                apply_butterfly_soa_simd(re + start, im + start,
                                         &tw.soa_re[tw.table_offsets[layer_idx]],
                                         &tw.soa_im[tw.table_offsets[layer_idx]],
                                         half, invert);
            }
        }
        return;
    }

    size_t half = n / 2;
    run_fft_soa(re, im, half, tw, invert);
    run_fft_soa(re + half, im + half, half, tw, invert);

    size_t t_idx = std::bit_width(n) - 1 - 3;
    apply_butterfly_soa_simd(re, im, &tw.soa_re[tw.table_offsets[t_idx]],
                             &tw.soa_im[tw.table_offsets[t_idx]], half, invert);
}
