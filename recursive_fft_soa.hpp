#pragma once

#include "types.hpp"
#include "base_fft.hpp"

class FFTRecursiveSoA : public FFTBase
{
public:
    using FFTBase::FFTBase;

    FFTLayout getLayout() const override { return FFTLayout::SoA; }
    void transform(SoAData &data, bool invert) override;

private:
    void run_fft_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t n, 
                                  const TwiddleData &tw, bool invert);
};
