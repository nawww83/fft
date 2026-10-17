#pragma once

#include "base_fft.hpp"

/**
 * @brief Итеративное БПФ (SoA) с ручными AVX2/FMA-ядрами.
 *
 * Отличия от FFTIterativeSoA:
 *   * бабочки, bit-reversal и нормировка считаются векторизованными
 *     ядрами из simd_kernels.hpp (4 бабочки за итерацию AVX2);
 *   * MSVC не автовекторизует этот код, поэтому выигрыш только от
 *     явных интринсиков.
 */
class FFTIterativeSoA_SIMD : public FFTBase
{
public:
    using FFTBase::FFTBase;

    FFTLayout getLayout() const override { return FFTLayout::SoA; }

    void transform(SoAData &data, bool invert) override;
};
