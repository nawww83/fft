#include "iterative_fft.hpp"

FFTIterative::FFTIterative(size_t max_n) : max_n(max_n)
{
    // Предварительный расчет таблиц поворотных коэффициентов (LUT)
    // Суммарный объем: N/4 + N/8 + ... + 2 + 1 ≈ N/2 элементов.
    full_table.reserve(max_n / 2);

    // Генерируем LUT начиная с len=8, так как уровни 2 и 4 оптимизированы вручную
    for (size_t len = 8; len <= max_n; len <<= 1)
    {
        table_offsets.push_back(full_table.size());

        const size_t half = len >> 1;
        const double angle_base = -2.0 * std::numbers::pi / len;

        for (size_t j = 0; j < half; ++j)
        {
            full_table.emplace_back(std::polar(1.0, j * angle_base));
        }
    }
}

void FFTIterative::transform(ComplexVec &v, bool invert) const {
    if (v.size() <= 1) return;
    
    if (invert) {
        execute_transform<true>(v);
    } else {
        execute_transform<false>(v);
    }
}
