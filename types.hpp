#pragma once

#include <vector>
#include <complex>

using RealVec = std::vector<double>;
using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;

// Исправляем ключевое слово restrict для MSVC
#ifdef _MSC_VER
    #define RESTRICT __restrict
#else
    #define RESTRICT __restrict__
#endif