#include "iterative_fft_soa_simd.hpp"
#include "simd_kernels.hpp"

void FFTIterativeSoA_SIMD::transform(SoAData &data, bool invert)
{
    const size_t n = data.re.size();
    if (n <= 1)
        return;

    f64 *RESTRICT re = data.re.data();
    f64 *RESTRICT im = data.im.data();

    fftsimd::bitrev_soa(re, im, m_swaps->get_for_n(n));

    if (invert)
        fftsimd::layers_soa<true>(re, im, n, *m_twiddles);
    else
        fftsimd::layers_soa<false>(re, im, n, *m_twiddles);

    if (invert)
        fftsimd::scale_soa(re, im, n, 1.0 / static_cast<f64>(n));
}
