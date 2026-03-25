
#pragma once

#include <vector>
#include <complex>

using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;


class FFTIterative {
public:
    explicit FFTIterative(size_t max_n);

    void transform(std::vector<Complex>& v, bool invert = false) const;

private:
    size_t max_n;
    std::vector<Complex> table;
};
