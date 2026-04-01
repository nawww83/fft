#include "iterative_fft.hpp"

FFTIterative::FFTIterative(size_t max_n) : max_n(max_n) {
    // Резервируем место под прямые и обратные коэффициенты
    full_table.reserve(max_n); 

    for (size_t len = 8; len <= max_n; len <<= 1) {
        table_offsets.push_back(full_table.size());
        const size_t half = len >> 1;
        const f64 angle_step = -2.0 * std::numbers::pi / len; // Выносим общую часть

        for (size_t j = 0; j < half; ++j) {
            f64 angle = angle_step * j;
            // Явный конструктор: проще для оптимизатора sincos
            full_table.emplace_back(std::cos(angle), std::sin(angle));
        }
    }

    // Предпосчитываем пары для всех возможных размеров n = 2^k
    for (size_t n = 2; n <= max_n; n <<= 1) {
        std::vector<SwapPair> pairs;
        int log2n = std::countr_zero(n);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t j = utils::fast_bit_reverse(i, log2n);
            if (i < j) {
                pairs.push_back({i, j});
            }
        }
        rev_tables.push_back(std::move(pairs));
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
