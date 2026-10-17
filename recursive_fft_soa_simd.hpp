#pragma once

#include "base_fft.hpp"

/**
 * @brief Рекурсивное БПФ (SoA) с ручными AVX2/FMA-ядрами.
 *
 * Ниже порога kBaseCase рекурсия сворачивается в итеративный проход по
 * всем слоям (len = 2..n) — это экономит накладные расходы вызовов на
 * мелких подмассивах, а сами слои выполняются общими SIMD-ядрами.
 */
class FFTRecursiveSoA_SIMD : public FFTBase
{
public:
    using FFTBase::FFTBase;

    FFTLayout getLayout() const override { return FFTLayout::SoA; }

    void transform(SoAData &data, bool invert) override;

    /** Порог, ниже которого рекурсия заменяется итеративным проходом. */
    static constexpr size_t kBaseCase = 32;

private:
    void run(f64 *RESTRICT re, f64 *RESTRICT im, size_t n,
             const TwiddleData &tw, bool invert) const;
};
