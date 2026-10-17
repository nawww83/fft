#pragma once
// =====================================================================
//  Ручные AVX2/FMA-ядра для БПФ (radix-2 DIT, in-place).
//
//  Мотивация: MSVC при /arch:AVX2 /fp:fast /Ox практически не
//  автовекторизует циклы бабочек — в ассемблере сплошные vmovsd /
//  vmulsd / vaddsd (скалярный SSE2), т.е. ~0.8 FLOPS на такт вместо
//  теоретических ~16 для AVX2+FMA. Поэтому ядра написаны вручную на
//  интринсиках, а скалярные версии оставлены как fallback (не-AVX2
//  сборки) и как обработка «хвостов» блоков, не кратных 4.
//
//  Раскладки памяти:
//    SoA : re[] и im[] — отдельные массивы; одна AVX2-итерация
//          обрабатывает 4 бабочки (4 x f64).
//    AoS : std::complex<f64> = [re, im] подряд; одна AVX2-итерация
//          обрабатывает 2 бабочки. Дайэринтерлив не нужен —
//          комплексное умножение считается через
//          permute_pd + mul_pd + addsub_pd.
//
//  Замечание про FMA: MSVC не определяет __FMA__, но FMA-интринсики
//  при /arch:AVX2 доступны, поэтому дополнительно проверяем _MSC_VER.
// =====================================================================

#include "types.hpp"

#include <bit>
#include <cstddef>
#include <utility>
#include <vector>

#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
    #define FFT_SIMD_AVX2 1
    #include <immintrin.h>
#else
    #define FFT_SIMD_AVX2 0
#endif

namespace fftsimd
{

// =====================================================================
//  Скалярные ядра: fallback для не-AVX2 сборок и «хвосты» SIMD-петель
// =====================================================================

inline void butterfly_soa_scalar(f64 *RESTRICT re, f64 *RESTRICT im, size_t half,
                                 const f64 *RESTRICT tw_re, const f64 *RESTRICT tw_im,
                                 bool invert) noexcept
{
    const f64 s = invert ? -1.0 : 1.0;
    for (size_t j = 0; j < half; ++j)
    {
        const f64 wr = tw_re[j];
        const f64 wi = s * tw_im[j];
        const f64 hr = re[j + half], hi = im[j + half];
        const f64 tr = hr * wr - hi * wi;
        const f64 ti = hr * wi + hi * wr;
        const f64 lr = re[j], li = im[j];
        re[j + half] = lr - tr;
        im[j + half] = li - ti;
        re[j] = lr + tr;
        im[j] = li + ti;
    }
}

inline void butterfly_aos_scalar(Complex64 *RESTRICT lo, Complex64 *RESTRICT hi,
                                 const Complex64 *RESTRICT tw, size_t half,
                                 bool invert) noexcept
{
    for (size_t j = 0; j < half; ++j)
    {
        const Complex64 w = invert ? Complex64{tw[j].real(), -tw[j].imag()} : tw[j];
        const Complex64 u = lo[j];
        const Complex64 t = w * hi[j];
        lo[j] = u + t;
        hi[j] = u - t;
    }
}

// =====================================================================
//  Слой len >= 8: основное ядро бабочки (здесь проходит ~90% времени)
// =====================================================================

/**
 * @brief Один слой БПФ для SoA-раскладки, half = len/2.
 *        Обрабатывает 4 бабочки за итерацию.
 */
template <bool Invert>
inline void butterfly_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t half,
                          const f64 *RESTRICT tw_re, const f64 *RESTRICT tw_im) noexcept
{
    size_t j = 0;
#if FFT_SIMD_AVX2
    const __m256d sign = _mm256_set1_pd(-0.0);
    for (; j + 4 <= half; j += 4)
    {
        const __m256d wr = _mm256_loadu_pd(tw_re + j);
        __m256d wi = _mm256_loadu_pd(tw_im + j);
        if constexpr (Invert)
            wi = _mm256_xor_pd(wi, sign); // сопряжение поворотного множителя

        const __m256d hr = _mm256_loadu_pd(re + j + half);
        const __m256d hi = _mm256_loadu_pd(im + j + half);

        // tr = hr*wr - hi*wi ; ti = hr*wi + hi*wr
        const __m256d tr = _mm256_fmsub_pd(hr, wr, _mm256_mul_pd(hi, wi));
        const __m256d ti = _mm256_fmadd_pd(hr, wi, _mm256_mul_pd(hi, wr));

        const __m256d lr = _mm256_loadu_pd(re + j);
        const __m256d li = _mm256_loadu_pd(im + j);
        _mm256_storeu_pd(re + j + half, _mm256_sub_pd(lr, tr));
        _mm256_storeu_pd(im + j + half, _mm256_sub_pd(li, ti));
        _mm256_storeu_pd(re + j, _mm256_add_pd(lr, tr));
        _mm256_storeu_pd(im + j, _mm256_add_pd(li, ti));
    }
#endif
    if (j < half)
        butterfly_soa_scalar(re + j, im + j, half - j, tw_re + j, tw_im + j, Invert);
}

/**
 * @brief Один слой БПФ для AoS-раскладки, half = len/2.
 *        Обрабатывает 2 бабочки за итерацию.
 *
 *        Комплексное умножение без дайэринтерлива:
 *          V   = [hr0, hi0, hr1, hi1]
 *          Wr  = [wr0, wr0, wr1, wr1]   (movedup)
 *          Wi  = [wi0, wi0, wi1, wi1]   (permute_pd 0b1111)
 *          Vs  = [hi0, hr0, hi1, hr1]   (permute_pd 0b0101)
 *          T   = addsub(Wr*V, Wi*Vs) = [tr0, ti0, tr1, ti1]
 */
template <bool Invert>
inline void butterfly_aos(Complex64 *RESTRICT lo, Complex64 *RESTRICT hi,
                          const Complex64 *RESTRICT tw, size_t half) noexcept
{
    size_t j = 0;
#if FFT_SIMD_AVX2
    const __m256d sign = _mm256_set1_pd(-0.0);
    const f64 *tw_d = reinterpret_cast<const f64 *>(tw);
    f64 *lo_d = reinterpret_cast<f64 *>(lo);
    f64 *hi_d = reinterpret_cast<f64 *>(hi);

    for (; j + 2 <= half; j += 2)
    {
        const __m256d w = _mm256_loadu_pd(tw_d + 2 * j); // [wr0, wi0, wr1, wi1]
        __m256d wi = _mm256_permute_pd(w, 0b1111);       // [wi0, wi0, wi1, wi1]
        if constexpr (Invert)
            wi = _mm256_xor_pd(wi, sign);

        const __m256d wr = _mm256_movedup_pd(w);         // [wr0, wr0, wr1, wr1]
        const __m256d v = _mm256_loadu_pd(hi_d + 2 * j);
        const __m256d u = _mm256_loadu_pd(lo_d + 2 * j);
        const __m256d vs = _mm256_permute_pd(v, 0b0101); // [hi0, hr0, hi1, hr1]

        // [wr*hr - wi*hi, wr*hi + wi*hr, ...] == [tr, ti, ...]
        const __m256d t = _mm256_addsub_pd(_mm256_mul_pd(wr, v), _mm256_mul_pd(wi, vs));

        _mm256_storeu_pd(lo_d + 2 * j, _mm256_add_pd(u, t));
        _mm256_storeu_pd(hi_d + 2 * j, _mm256_sub_pd(u, t));
    }
#endif
    if (j < half)
        butterfly_aos_scalar(lo + j, hi + j, tw + j, half - j, Invert);
}

// =====================================================================
//  Слой len == 2: бабочка без тригонометрии
// =====================================================================

inline void radix2_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t n) noexcept
{
    size_t i = 0;
#if FFT_SIMD_AVX2
    for (; i + 4 <= n; i += 4)
    {
        // [x0,x1,x2,x3] -> [x0+x1, x0-x1, x2+x3, x2-x3]
        const __m256d x = _mm256_loadu_pd(re + i);
        const __m256d sw = _mm256_permute4x64_pd(x, 0xB1); // [x1,x0,x3,x2]
        _mm256_storeu_pd(re + i, _mm256_blend_pd(_mm256_add_pd(x, sw),
                                                 _mm256_sub_pd(sw, x), 0b1010));

        const __m256d y = _mm256_loadu_pd(im + i);
        const __m256d sw2 = _mm256_permute4x64_pd(y, 0xB1);
        _mm256_storeu_pd(im + i, _mm256_blend_pd(_mm256_add_pd(y, sw2),
                                                 _mm256_sub_pd(sw2, y), 0b1010));
    }
#endif
    for (; i + 2 <= n; i += 2)
    {
        const f64 a = re[i], b = re[i + 1];
        const f64 c = im[i], d = im[i + 1];
        re[i] = a + b; re[i + 1] = a - b;
        im[i] = c + d; im[i + 1] = c - d;
    }
}

inline void radix2_aos(Complex64 *RESTRICT d, size_t n) noexcept
{
    size_t i = 0;
#if FFT_SIMD_AVX2
    f64 *p = reinterpret_cast<f64 *>(d);
    for (; i + 2 <= n; i += 2)
    {
        // [r0,i0,r1,i1] -> [r0+r1, i0+i1, r0-r1, i0-i1]
        const __m256d v = _mm256_loadu_pd(p + 2 * i);
        const __m256d sw = _mm256_permute4x64_pd(v, 0b01001110); // [r1,i1,r0,i0]
        _mm256_storeu_pd(p + 2 * i, _mm256_blend_pd(_mm256_add_pd(v, sw),
                                                    _mm256_sub_pd(sw, v), 0b1100));
    }
#endif
    for (; i + 2 <= n; i += 2)
    {
        const Complex64 u = d[i], t = d[i + 1];
        d[i] = u + t;
        d[i + 1] = u - t;
    }
}

// =====================================================================
//  Слой len == 4: фиксированные поворотные множители {1, ∓i}
// =====================================================================

template <bool Invert>
inline void radix4_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t n) noexcept
{
    // Скалярно: в SoA перестановка w = ∓i требует кросс-лейновых выборок,
    // а сам слой линеен по n и не является горячим.
    const f64 s = Invert ? -1.0 : 1.0;
    for (size_t i = 0; i + 4 <= n; i += 4)
    {
        const f64 r0 = re[i], r1 = re[i + 1], r2 = re[i + 2], r3 = re[i + 3];
        const f64 i0 = im[i], i1 = im[i + 1], i2 = im[i + 2], i3 = im[i + 3];

        // пара (0,2) с w = 1
        re[i] = r0 + r2; re[i + 2] = r0 - r2;
        im[i] = i0 + i2; im[i + 2] = i0 - i2;

        // пара (1,3) с w = ∓i
        const f64 tr = s * i3;
        const f64 ti = -s * r3;
        re[i + 1] = r1 + tr; re[i + 3] = r1 - tr;
        im[i + 1] = i1 + ti; im[i + 3] = i1 - ti;
    }
}

template <bool Invert>
inline void radix4_aos(Complex64 *RESTRICT d, size_t n) noexcept
{
    size_t i = 0;
#if FFT_SIMD_AVX2
    f64 *p = reinterpret_cast<f64 *>(d);
    // forward: c3*(0-i) = [ i3, -r3]  -> знак только на последнем элементе
    // inverse: c3*(0+i) = [-i3,  r3]  -> знак на предпоследнем
    const __m256d m = Invert ? _mm256_set_pd(0.0, -0.0, 0.0, 0.0)
                             : _mm256_set_pd(-0.0, 0.0, 0.0, 0.0);
    for (; i + 4 <= n; i += 4)
    {
        const __m256d v0 = _mm256_loadu_pd(p + 2 * i);     // c0, c1
        const __m256d v1 = _mm256_loadu_pd(p + 2 * i + 4); // c2, c3
        // T = [c2, c3 * (∓i)] = [r2, i2, i3, -r3]
        const __m256d t = _mm256_xor_pd(_mm256_permute4x64_pd(v1, 0xB4), m);
        _mm256_storeu_pd(p + 2 * i, _mm256_add_pd(v0, t));
        _mm256_storeu_pd(p + 2 * i + 4, _mm256_sub_pd(v0, t));
    }
#endif
    const f64 s = Invert ? -1.0 : 1.0;
    for (; i + 4 <= n; i += 4)
    {
        const Complex64 u0 = d[i], u1 = d[i + 1];
        const Complex64 t0 = d[i + 2];
        const Complex64 t1 = {s * d[i + 3].imag(), -s * d[i + 3].real()};
        d[i] = u0 + t0; d[i + 2] = u0 - t0;
        d[i + 1] = u1 + t1; d[i + 3] = u1 - t1;
    }
}

// =====================================================================
//  Полный набор слоёв DIT для массива длины n (len = 2, 4, 8, ..., n)
//
//  Кэш-блокирование. Слой len комбинирует элементы только внутри
//  непрерывного участка памяти длиной len, поэтому все слои с len <= B
//  можно выполнить внутри каждого B-элементного блока, пока он resident
//  в L1. Это заменяет log2(B) полных проходов по памяти одним.
//  Верхние слои (len > B) по-прежнему стримят массив целиком.
//
//  Для N = 2^20 и B = 512: 12 проходов вместо 20 (~1.6x меньше трафика).
//  Дальнейшее увеличение B упирается в размер L1.
// =====================================================================

#ifndef FFT_SIMD_BLOCK
// Размер кэш-блока: данные блока (2*B*8 байт) + его поворотные множители
// должны укладываться в L1 (32 КБ). B = 512 -> ~8 КБ + ~8 КБ.
#define FFT_SIMD_BLOCK 512
#endif

inline constexpr size_t kBlockSize = FFT_SIMD_BLOCK;

/** @brief Слои len = first_len .. n одним непрерывным проходом. */
template <bool Invert>
inline void layers_range_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t n,
                             const TwiddleData &tw, size_t first_len) noexcept
{
    for (size_t len = first_len; len <= n; len <<= 1)
    {
        const size_t half = len >> 1;
        const size_t t_idx = static_cast<size_t>(std::bit_width(len)) - 4;
        const f64 *tw_re = &tw.soa_re[tw.table_offsets[t_idx]];
        const f64 *tw_im = &tw.soa_im[tw.table_offsets[t_idx]];
        for (size_t start = 0; start < n; start += len)
        {
            if constexpr (Invert)
                butterfly_soa<true>(re + start, im + start, half, tw_re, tw_im);
            else
                butterfly_soa<false>(re + start, im + start, half, tw_re, tw_im);
        }
    }
}

template <bool Invert>
inline void layers_range_aos(Complex64 *RESTRICT d, size_t n,
                             const TwiddleData &tw, size_t first_len) noexcept
{
    for (size_t len = first_len; len <= n; len <<= 1)
    {
        const size_t half = len >> 1;
        const size_t t_idx = static_cast<size_t>(std::bit_width(len)) - 4;
        const Complex64 *t = &tw.aos[tw.table_offsets[t_idx]];
        for (size_t start = 0; start < n; start += len)
        {
            if constexpr (Invert)
                butterfly_aos<true>(d + start, d + start + half, t, half);
            else
                butterfly_aos<false>(d + start, d + start + half, t, half);
        }
    }
}

/** @brief Все слои (len = 2..n) для n, укладывающегося в кэш-блок. */
template <bool Invert>
inline void layers_block_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t n,
                             const TwiddleData &tw) noexcept
{
    radix2_soa(re, im, n);
    if (n < 4)
        return;
    if constexpr (Invert)
        radix4_soa<true>(re, im, n);
    else
        radix4_soa<false>(re, im, n);
    if (n < 8)
        return;
    layers_range_soa<Invert>(re, im, n, tw, 8);
}

template <bool Invert>
inline void layers_block_aos(Complex64 *RESTRICT d, size_t n,
                             const TwiddleData &tw) noexcept
{
    radix2_aos(d, n);
    if (n < 4)
        return;
    if constexpr (Invert)
        radix4_aos<true>(d, n);
    else
        radix4_aos<false>(d, n);
    if (n < 8)
        return;
    layers_range_aos<Invert>(d, n, tw, 8);
}

/** @brief Полное ДПФ-ядро DIT для массива длины n (n — степень двойки). */
template <bool Invert>
inline void layers_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t n,
                       const TwiddleData &tw) noexcept
{
    if (n <= kBlockSize)
    {
        layers_block_soa<Invert>(re, im, n, tw);
        return;
    }
    for (size_t start = 0; start < n; start += kBlockSize)
        layers_block_soa<Invert>(re + start, im + start, kBlockSize, tw);
    layers_range_soa<Invert>(re, im, n, tw, kBlockSize << 1);
}

template <bool Invert>
inline void layers_aos(Complex64 *RESTRICT d, size_t n, const TwiddleData &tw) noexcept
{
    if (n <= kBlockSize)
    {
        layers_block_aos<Invert>(d, n, tw);
        return;
    }
    for (size_t start = 0; start < n; start += kBlockSize)
        layers_block_aos<Invert>(d + start, kBlockSize, tw);
    layers_range_aos<Invert>(d, n, tw, kBlockSize << 1);
}

// =====================================================================
//  Bit-reversal.
//
//  Перестановка — это набор независимых транспозиций с непересекающимися
//  носителями, поэтому их можно выполнять в любом порядке и пачками.
//  Перестановка «в один проход» упирается в latency DRAM: каждый обмен
//  ждёт загрузку случайной строки кэша. Поэтому:
//    * читаем/пишем пачками по 4 транспозиции (16 независимых загрузок
//      в полёте -> работает memory-level parallelism);
//    * заранее (на 3 пачки вперёд) предзагружаем «далёкие» адреса J.
//  Замер: 2.4..3.1x против наивного цикла std::swap на N = 2^20..2^22.
// =====================================================================

inline void bitrev_aos(Complex64 *RESTRICT p, const std::vector<SwapPair> &pairs) noexcept
{
    f64 *pd = reinterpret_cast<f64 *>(p);
    const size_t m = pairs.size();
    size_t k = 0;
#if FFT_SIMD_AVX2
    for (; k + 4 <= m; k += 4)
    {
        if (k + 16 <= m)
        {
            _mm_prefetch(reinterpret_cast<const char *>(p + pairs[k + 12].j), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(p + pairs[k + 13].j), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(p + pairs[k + 14].j), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(p + pairs[k + 15].j), _MM_HINT_T0);
        }
        const __m128d a0 = _mm_loadu_pd(pd + 2 * (size_t)pairs[k + 0].i);
        const __m128d a1 = _mm_loadu_pd(pd + 2 * (size_t)pairs[k + 1].i);
        const __m128d a2 = _mm_loadu_pd(pd + 2 * (size_t)pairs[k + 2].i);
        const __m128d a3 = _mm_loadu_pd(pd + 2 * (size_t)pairs[k + 3].i);
        const __m128d b0 = _mm_loadu_pd(pd + 2 * (size_t)pairs[k + 0].j);
        const __m128d b1 = _mm_loadu_pd(pd + 2 * (size_t)pairs[k + 1].j);
        const __m128d b2 = _mm_loadu_pd(pd + 2 * (size_t)pairs[k + 2].j);
        const __m128d b3 = _mm_loadu_pd(pd + 2 * (size_t)pairs[k + 3].j);
        _mm_storeu_pd(pd + 2 * (size_t)pairs[k + 0].i, b0);
        _mm_storeu_pd(pd + 2 * (size_t)pairs[k + 1].i, b1);
        _mm_storeu_pd(pd + 2 * (size_t)pairs[k + 2].i, b2);
        _mm_storeu_pd(pd + 2 * (size_t)pairs[k + 3].i, b3);
        _mm_storeu_pd(pd + 2 * (size_t)pairs[k + 0].j, a0);
        _mm_storeu_pd(pd + 2 * (size_t)pairs[k + 1].j, a1);
        _mm_storeu_pd(pd + 2 * (size_t)pairs[k + 2].j, a2);
        _mm_storeu_pd(pd + 2 * (size_t)pairs[k + 3].j, a3);
    }
#endif
    for (; k < m; ++k)
        std::swap(p[pairs[k].i], p[pairs[k].j]);
}

inline void bitrev_soa(f64 *RESTRICT re, f64 *RESTRICT im,
                       const std::vector<SwapPair> &pairs) noexcept
{
    const size_t m = pairs.size();
    size_t k = 0;
    for (; k + 4 <= m; k += 4)
    {
#if FFT_SIMD_AVX2
        if (k + 16 <= m)
        {
            _mm_prefetch(reinterpret_cast<const char *>(re + pairs[k + 12].j), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(im + pairs[k + 12].j), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(re + pairs[k + 14].j), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(im + pairs[k + 14].j), _MM_HINT_T0);
        }
#endif
        // Фаза чтения: 16 независимых загрузок в полёте
        const f64 ar0 = re[pairs[k + 0].i], ai0 = im[pairs[k + 0].i];
        const f64 ar1 = re[pairs[k + 1].i], ai1 = im[pairs[k + 1].i];
        const f64 ar2 = re[pairs[k + 2].i], ai2 = im[pairs[k + 2].i];
        const f64 ar3 = re[pairs[k + 3].i], ai3 = im[pairs[k + 3].i];
        const f64 br0 = re[pairs[k + 0].j], bi0 = im[pairs[k + 0].j];
        const f64 br1 = re[pairs[k + 1].j], bi1 = im[pairs[k + 1].j];
        const f64 br2 = re[pairs[k + 2].j], bi2 = im[pairs[k + 2].j];
        const f64 br3 = re[pairs[k + 3].j], bi3 = im[pairs[k + 3].j];
        // Фаза записи
        re[pairs[k + 0].i] = br0; im[pairs[k + 0].i] = bi0;
        re[pairs[k + 1].i] = br1; im[pairs[k + 1].i] = bi1;
        re[pairs[k + 2].i] = br2; im[pairs[k + 2].i] = bi2;
        re[pairs[k + 3].i] = br3; im[pairs[k + 3].i] = bi3;
        re[pairs[k + 0].j] = ar0; im[pairs[k + 0].j] = ai0;
        re[pairs[k + 1].j] = ar1; im[pairs[k + 1].j] = ai1;
        re[pairs[k + 2].j] = ar2; im[pairs[k + 2].j] = ai2;
        re[pairs[k + 3].j] = ar3; im[pairs[k + 3].j] = ai3;
    }
    for (; k < m; ++k)
    {
        const SwapPair pr = pairs[k];
        std::swap(re[pr.i], re[pr.j]);
        std::swap(im[pr.i], im[pr.j]);
    }
}

// =====================================================================
//  Нормировка (только обратное преобразование)
// =====================================================================

inline void scale_soa(f64 *RESTRICT re, f64 *RESTRICT im, size_t n, f64 inv_n) noexcept
{
    size_t i = 0;
#if FFT_SIMD_AVX2
    const __m256d s = _mm256_set1_pd(inv_n);
    for (; i + 4 <= n; i += 4)
    {
        _mm256_storeu_pd(re + i, _mm256_mul_pd(_mm256_loadu_pd(re + i), s));
        _mm256_storeu_pd(im + i, _mm256_mul_pd(_mm256_loadu_pd(im + i), s));
    }
#endif
    for (; i < n; ++i)
    {
        re[i] *= inv_n;
        im[i] *= inv_n;
    }
}

inline void scale_aos(Complex64 *RESTRICT d, size_t n, f64 inv_n) noexcept
{
    size_t i = 0;
#if FFT_SIMD_AVX2
    f64 *p = reinterpret_cast<f64 *>(d);
    const __m256d s = _mm256_set1_pd(inv_n);
    for (; i + 2 <= n; i += 2)
        _mm256_storeu_pd(p + 2 * i, _mm256_mul_pd(_mm256_loadu_pd(p + 2 * i), s));
#endif
    for (; i < n; ++i)
        d[i] *= inv_n;
}

} // namespace fftsimd
