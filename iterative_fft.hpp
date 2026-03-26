
#pragma once

#include "types.hpp"

class FFTIterative {
public:
    explicit FFTIterative(size_t max_n);

    void transform(ComplexVec& v, bool invert = false) const;

private:
    size_t max_n;
    ComplexVec table;
};
