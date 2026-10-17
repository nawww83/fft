#pragma once

#include "base_fft.hpp"

/**
 * @brief Итеративное БПФ (AoS) с ручными AVX2/FMA-ядрами.
 *
 * Дайэринтерлив не выполняется: комплексное умножение считается прямо
 * в AoS-раскладке через permute_pd + mul_pd + addsub_pd (2 бабочки за
 * итерацию AVX2).
 */
class FFTIterativeAoS_SIMD : public FFTBase
{
public:
    using FFTBase::FFTBase;

    FFTLayout getLayout() const override { return FFTLayout::AoS; }

    void transform(AoSData &data, bool invert) override;
};
