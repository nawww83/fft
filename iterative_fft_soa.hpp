#pragma once

#include "types.hpp"

class FFTIterativeSoA {
    RealVec table_re;
    RealVec table_im;
    std::vector<size_t> table_offsets;
    struct SwapPair { u32 i, j; };
    std::vector<std::vector<SwapPair>> rev_tables;
    mutable RealVec re_buffer;
    mutable RealVec im_buffer;

public:
    FFTIterativeSoA(size_t max_n);

    void transform(ComplexVec& v, bool invert) const;

private:
    void execute_soa(f64* RESTRICT re, f64* RESTRICT im, size_t n, bool invert) const;

    static void apply_layer_soa(f64* RESTRICT re, f64* RESTRICT im, 
                            size_t half, 
                            const f64* RESTRICT tw_re, 
                            const f64* RESTRICT tw_im, 
                            bool invert);

}; // FFTIterative