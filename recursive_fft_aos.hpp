#pragma once

#include "types.hpp"
#include "base_fft.hpp"

class FFTRecursiveAoS : public FFTBase
{
public:
    using FFTBase::FFTBase;

    FFTLayout getLayout() const override { return FFTLayout::AoS; }
    void transform(AoSData &data, bool invert) override;

private:
    void run_fft_inplace_fwd(Complex64 *data, size_t n,
                             const TwiddleData &twiddles);

    void run_fft_inplace_inv(Complex64 *data, size_t n,
                             const TwiddleData &twiddles);
};
