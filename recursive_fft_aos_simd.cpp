#include "recursive_fft_aos_simd.hpp"
#include "simd_kernels.hpp"

#include <bit>

void FFTRecursiveAoS_SIMD::transform(AoSData &data, bool invert)
{
    const size_t n = data.buffer.size();
    if (n <= 1)
        return;

    Complex64 *RESTRICT p = data.buffer.data();

    fftsimd::bitrev_aos(p, m_swaps->get_for_n(n));
    run(p, n, *m_twiddles, invert);

    if (invert)
        fftsimd::scale_aos(p, n, 1.0 / static_cast<f64>(n));
}

void FFTRecursiveAoS_SIMD::run(Complex64 *RESTRICT d, size_t n,
                               const TwiddleData &tw, bool invert) const
{
    if (n <= kBaseCase)
    {
        if (invert)
            fftsimd::layers_aos<true>(d, n, tw);
        else
            fftsimd::layers_aos<false>(d, n, tw);
        return;
    }

    const size_t half = n >> 1;
    run(d, half, tw, invert);
    run(d + half, half, tw, invert);

    // Сборка уровня: len == n, индекс таблицы = log2(n) - 3 - 1
    const size_t t_idx = static_cast<size_t>(std::bit_width(n)) - 4;
    const Complex64 *t = &tw.aos[tw.table_offsets[t_idx]];
    if (invert)
        fftsimd::butterfly_aos<true>(d, d + half, t, half);
    else
        fftsimd::butterfly_aos<false>(d, d + half, t, half);
}
