#include "iterative_fft_aos_simd.hpp"
#include "simd_kernels.hpp"

void FFTIterativeAoS_SIMD::transform(AoSData &data, bool invert)
{
    const size_t n = data.buffer.size();
    if (n <= 1)
        return;

    Complex64 *RESTRICT p = data.buffer.data();

    fftsimd::bitrev_aos(p, m_swaps->get_for_n(n));

    if (invert)
        fftsimd::layers_aos<true>(p, n, *m_twiddles);
    else
        fftsimd::layers_aos<false>(p, n, *m_twiddles);

    if (invert)
        fftsimd::scale_aos(p, n, 1.0 / static_cast<f64>(n));
}
