#include "recursive_fft_soa_simd.hpp"
#include "simd_kernels.hpp"

#include <bit>

void FFTRecursiveSoA_SIMD::transform(SoAData &data, bool invert)
{
    const size_t n = data.re.size();
    if (n <= 1)
        return;

    f64 *RESTRICT re = data.re.data();
    f64 *RESTRICT im = data.im.data();

    fftsimd::bitrev_soa(re, im, m_swaps->get_for_n(n));
    run(re, im, n, *m_twiddles, invert);

    if (invert)
        fftsimd::scale_soa(re, im, n, 1.0 / static_cast<f64>(n));
}

void FFTRecursiveSoA_SIMD::run(f64 *RESTRICT re, f64 *RESTRICT im, size_t n,
                               const TwiddleData &tw, bool invert) const
{
    if (n <= kBaseCase)
    {
        if (invert)
            fftsimd::layers_soa<true>(re, im, n, tw);
        else
            fftsimd::layers_soa<false>(re, im, n, tw);
        return;
    }

    const size_t half = n >> 1;
    run(re, im, half, tw, invert);
    run(re + half, im + half, half, tw, invert);

    // Сборка уровня: len == n, индекс таблицы = log2(n) - 3 - 1
    const size_t t_idx = static_cast<size_t>(std::bit_width(n)) - 4;
    const f64 *tw_re = &tw.soa_re[tw.table_offsets[t_idx]];
    const f64 *tw_im = &tw.soa_im[tw.table_offsets[t_idx]];
    if (invert)
        fftsimd::butterfly_soa<true>(re, im, half, tw_re, tw_im);
    else
        fftsimd::butterfly_soa<false>(re, im, half, tw_re, tw_im);
}
