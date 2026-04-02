#include "iterative_fft_soa.hpp"
#include <numbers>

void FFTIterativeSoA::execute_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t n, bool invert) const
{
    if (n < 2)
        return;
    // --- СЛОЙ len=2 (Butterfly без тригонометрии) ---
    for (size_t i = 0; i < n; i += 2)
    {
        f64 r0 = re[i], r1 = re[i + 1];
        f64 i0 = im[i], i1 = im[i + 1];
        re[i] = r0 + r1;
        re[i + 1] = r0 - r1;
        im[i] = i0 + i1;
        im[i + 1] = i0 - i1;
    }
    if (n < 4)
        goto scaling;
    // --- СЛОЙ len=4 (Butterfly с фиксированными w=1 и w=±i) ---
    for (size_t i = 0; i < n; i += 4)
    {
        // Пара 0-2 (w=1)
        f64 r0 = re[i], r2 = re[i + 2];
        f64 i0 = im[i], i2 = im[i + 2];
        re[i] = r0 + r2;
        re[i + 2] = r0 - r2;
        im[i] = i0 + i2;
        im[i + 2] = i0 - i2;

        // Пара 1-3 (w=±i)
        f64 r1 = re[i + 1], r3 = re[i + 3];
        f64 i1 = im[i + 1], i3 = im[i + 3];

        // Для прямого: (r3+i*i3)*(0-i) = i3 - i*r3
        // Для обратного: (r3+i*i3)*(0+i) = -i3 + i*r3
        f64 tr = invert ? -i3 : i3;
        f64 ti = invert ? r3 : -r3;

        re[i + 1] = r1 + tr;
        re[i + 3] = r1 - tr;
        im[i + 1] = i1 + ti;
        im[i + 3] = i1 - ti;
    }
    // --- ОСНОВНЫЕ СЛОИ (len >= 8) ---
    {
        size_t t_idx = 0; // Начинаем с первой записи в таблице (которая для len=8)
        for (size_t len = 8; len <= n; len <<= 1)
        {
            const size_t half = len >> 1;
            const f64 *tw_re = &m_twiddles->soa_re[m_twiddles->table_offsets[t_idx]];
            const f64 *tw_im = &m_twiddles->soa_im[m_twiddles->table_offsets[t_idx]];
            t_idx++;

            for (size_t start = 0; start < n; start += len)
            {
                if (invert)
                    apply_layer_soa_inv(re + start, im + start, half, tw_re, tw_im);
                else
                    apply_layer_soa_fwd(re + start, im + start, half, tw_re, tw_im);
            }
        }
    }

scaling:
    if (invert)
    {
        f64 inv_n = 1.0 / static_cast<f64>(n);
        for (size_t i = 0; i < n; ++i)
        {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

// Прямое преобразование (Forward)
void FFTIterativeSoA::apply_layer_soa_fwd(f64 *RESTRICT re, f64 *RESTRICT im, size_t half,
                         const f64 *RESTRICT tw_re, const f64 *RESTRICT tw_im)
{
    for (size_t j = 0; j < half; ++j)
    {
        f64 wr = tw_re[j], wi = tw_im[j];
        f64 hr = re[j + half], hi = im[j + half];

        // Complex: (hr + i*hi) * (wr + i*wi)
        f64 tr = hr * wr - hi * wi;
        f64 ti = hr * wi + hi * wr;

        f64 lr = re[j], li = im[j];
        re[j + half] = lr - tr;
        im[j + half] = li - ti;
        re[j] = lr + tr;
        im[j] = li + ti;
    }
}

// Обратное преобразование (Inverse)
void FFTIterativeSoA::apply_layer_soa_inv(f64 *RESTRICT re, f64 *RESTRICT im, size_t half,
                         const f64 *RESTRICT tw_re, const f64 *RESTRICT tw_im)
{
    for (size_t j = 0; j < half; ++j)
    {
        f64 wr = tw_re[j], wi = -tw_im[j]; // Инверсия мнимой части
        f64 hr = re[j + half], hi = im[j + half];

        f64 tr = hr * wr - hi * wi;
        f64 ti = hr * wi + hi * wr;

        f64 lr = re[j], li = im[j];
        re[j + half] = lr - tr;
        im[j + half] = li - ti;
        re[j] = lr + tr;
        im[j] = li + ti;
    }
}
