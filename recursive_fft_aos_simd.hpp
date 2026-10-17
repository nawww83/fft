#pragma once

#include "base_fft.hpp"

/**
 * @brief Рекурсивное БПФ (AoS) с ручными AVX2/FMA-ядрами.
 *
 * Ниже порога kBaseCase рекурсия сворачивается в итеративный проход по
 * всем слоям (len = 2..n) — это экономит накладные расходы вызовов на
 * мелких подмассивах, а сами слои выполняются общими SIMD-ядрами.
 */
class FFTRecursiveAoS_SIMD : public FFTBase
{
public:
    using FFTBase::FFTBase;

    FFTLayout getLayout() const override { return FFTLayout::AoS; }

    void transform(AoSData &data, bool invert) override;

    /** Порог, ниже которого рекурсия заменяется итеративным проходом. */
    static constexpr size_t kBaseCase = 32;

private:
    void run(Complex64 *RESTRICT d, size_t n,
             const TwiddleData &tw, bool invert) const;
};
