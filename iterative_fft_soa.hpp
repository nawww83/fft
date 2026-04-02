#pragma once

#include "base_fft.hpp"
#include <numbers>

class FFTIterativeSoA : public FFTBase
{
public:
    using FFTBase::FFTBase;

    FFTLayout getType() const override { return FFTLayout::SoA; }

    void transform(SoAData &data, bool invert) override
    {
        const auto n = data.re.size();
        if (n <= 1)
            return;
        const auto &pairs = m_swaps->get_for_n(n);

        // Bit-reversal swap
        for (const auto &p : pairs)
        {
            std::swap(data.re[p.i], data.re[p.j]);
            std::swap(data.im[p.i], data.im[p.j]);
        }

        // Работаем с указателями на внутренние буферы
        f64 *RESTRICT re_ptr = data.re.data();
        f64 *RESTRICT im_ptr = data.im.data();

        // Вычисления (Ядро БПФ)
        execute_soa(re_ptr, im_ptr, n, invert);
    }

private:
    void execute_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t n, bool invert) const;
    static void apply_layer_soa_fwd(f64 *RESTRICT re, f64 *RESTRICT im, size_t half,
                                    const f64 *RESTRICT tw_re, const f64 *RESTRICT tw_im);
    static void apply_layer_soa_inv(f64 *RESTRICT re, f64 *RESTRICT im, size_t half,
                                    const f64 *RESTRICT tw_re, const f64 *RESTRICT tw_im);

};