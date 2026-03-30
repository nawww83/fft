#pragma once

#include <span>
#include "types.hpp"

class FFTRecursive {
public:
    /**
     * @param max_n Максимальный размер (степень двойки) для предвычисления таблиц.
     */
    explicit FFTRecursive(size_t max_n);

    /**
     * @brief Прямое или обратное БПФ (In-place).
     * @param v Вектор данных (изменяется на месте).
     * @param invert true для обратного преобразования.
     */
    void transform(ComplexVec& v, bool invert = false);

private:
    /**
     * @brief Рекурсивная сборка "снизу вверх" после bit-reversal.
     */
    void run_fft_inplace(std::span<Complex> data,
                         const ComplexVec& twiddles,
                         size_t stride);

    static ComplexVec generate_twiddles(size_t n, bool invert);

    size_t m_max_n;
    ComplexVec m_twiddles;  
    ComplexVec m_itwiddles; 
};
