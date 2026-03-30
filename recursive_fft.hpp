#pragma once

#include "types.hpp"

class FFTRecursive {
public:
    explicit FFTRecursive(size_t max_n);
    void transform(ComplexVec& v, bool invert = false);

private:
    void run_fft_inplace(Complex* data, size_t n, int level, const std::vector<ComplexVec>& all_twiddles);

    size_t m_max_n;
    // Пирамида таблиц: m_twiddles[0] для n=2, [1] для n=4, [2] для n=8...
    std::vector<ComplexVec> m_twiddle_levels;
    std::vector<ComplexVec> m_itwiddle_levels;
};
