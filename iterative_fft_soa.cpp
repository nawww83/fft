#include "iterative_fft_soa.hpp"
#include "bit_reverse.hpp"
#include <numbers>

FFTIterativeSoA::FFTIterativeSoA(size_t max_n)
{
    // 1. Предрасчет таблиц (SoA формат)
    for (size_t len = 8; len <= max_n; len <<= 1)
    {
        table_offsets.push_back(table_re.size());
        size_t half = len >> 1;
        f64 angle_step = -2.0 * std::numbers::pi / len;
        for (size_t j = 0; j < half; ++j)
        {
            f64 angle = angle_step * j;
            table_re.push_back(std::cos(angle));
            table_im.push_back(std::sin(angle));
        }
    }
    // 2. Предрасчет Bit-reversal
    for (size_t n = 2; n <= max_n; n <<= 1)
    {
        std::vector<SwapPair> pairs;
        int bits = std::countr_zero(n);
        for (uint32_t i = 0; i < n; ++i)
        {
            uint32_t j = utils::fast_bit_reverse(i, bits);
            if (i < j)
                pairs.push_back({i, j});
        }
        rev_tables.push_back(std::move(pairs));
    }
    // Сразу резервируем память под максимальный размер
    re_buffer.reserve(max_n);
    im_buffer.reserve(max_n);
}

void FFTIterativeSoA::transform(ComplexVec &v, bool invert) const
{
    size_t n = v.size();
    if (n <= 1)
        return;

    // AoS -> SoA
    re_buffer.resize(n);
    im_buffer.resize(n);

    // Работаем с указателями на внутренние буферы
    f64 *RESTRICT re = re_buffer.data();
    f64 *RESTRICT im = im_buffer.data();

    // 1. AoS -> SoA
    for (size_t i = 0; i < n; ++i)
    {
        re[i] = v[i].real();
        im[i] = v[i].imag();
    }

    // 2. Вычисления (Ядро БПФ)
    execute_soa(re, im, n, invert);

    // 3. SoA -> AoS
    for (size_t i = 0; i < n; ++i)
    {
        v[i] = Complex(re[i], im[i]);
    }
}

void FFTIterativeSoA::execute_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t n, bool invert) const
{
    const int log2n = std::countr_zero(n);
    const auto &pairs = rev_tables[log2n - 1];

    // 1. Bit-reversal
    for (const auto &p : pairs)
    {
        std::swap(re[p.i], re[p.j]);
        std::swap(im[p.i], im[p.j]);
    }

    // 2. Слой len=2
    for (size_t i = 0; i < n; i += 2)
    {
        f64 tr = re[i + 1], ti = im[i + 1];
        re[i + 1] = re[i] - tr;
        im[i + 1] = im[i] - ti;
        re[i] = re[i] + tr;
        im[i] = im[i] + ti;
    }

    // 3. Слой len=4
    for (size_t i = 0; i < n; i += 8)
    {
        for (size_t k = 0; k < 2; ++k)
        {
            size_t idx = i + k * 4;

            // Группа w=1 (индексы idx и idx+2)
            f64 r0 = re[idx], r2 = re[idx + 2];
            f64 i0 = im[idx], i2 = im[idx + 2];

            re[idx] = r0 + r2;
            re[idx + 2] = r0 - r2;
            im[idx] = i0 + i2;
            im[idx + 2] = i0 - i2;

            // Группа w=-i (индексы idx+1 и idx+3)
            f64 r1 = re[idx + 1], r3 = re[idx + 3];
            f64 i1 = im[idx + 1], i3 = im[idx + 3];

            f64 tr = (invert ? -i3 : i3);
            f64 ti = (invert ? r3 : -r3);

            re[idx + 1] = r1 + tr;
            re[idx + 3] = r1 - tr;
            im[idx + 1] = i1 + ti;
            im[idx + 3] = i1 - ti;
        }
    }

    // 4. Основной цикл (len >= 8)
    size_t t_idx = 0;
    for (size_t len = 8; len <= n; len <<= 1)
    {
        const size_t half = len >> 1;
        const f64 *RESTRICT tw_re = &table_re[table_offsets[t_idx]];
        const f64 *RESTRICT tw_im = &table_im[table_offsets[t_idx++]];
        for (size_t start = 0; start < n; start += len)
        {
            apply_layer_soa(re + start, im + start, half, tw_re, tw_im, invert);
        }
    }
    if (invert)
    {
        f64 inv_n = 1.0 / n;
        for (size_t i = 0; i < n; ++i)
        {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

void FFTIterativeSoA::apply_layer_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t half, const f64 *RESTRICT tw_re, const f64 *RESTRICT tw_im, bool invert)
{
    const f64 s = invert ? -1.0 : 1.0;

    f64* RESTRICT r_low = re;
    f64* RESTRICT i_low = im;
    f64* RESTRICT r_high = re + half;
    f64* RESTRICT i_high = im + half;

    #pragma clang loop vectorize(enable) interleave(enable)
    for (size_t j = 0; j < half; ++j) {
        const f64 wr = tw_re[j];
        const f64 wi = tw_im[j] * s;

        const f64 hr = r_high[j];
        const f64 hi = i_high[j];

        // Комплексное умножение: (hr + i*hi) * (wr + i*wi)
        f64 tr = hr * wr - hi * wi;
        f64 ti = hr * wi + hi * wr;

        f64 lr = r_low[j];
        f64 li = i_low[j];

        r_high[j] = lr - tr;
        i_high[j] = li - ti;
        r_low[j]  = lr + tr;
        i_low[j]  = li + ti;
    }
}
