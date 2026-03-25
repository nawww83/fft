#pragma once

#include <vector>
#include <complex>
#include <span>

using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;

class FFTRecursive {
public:
    /**
     * @param max_n Максимальный размер (степень двойки) для предвычисления таблиц.
     */
    explicit FFTRecursive(size_t max_n);

    /**
     * @brief Прямое или обратное БПФ.
     * @param v Вектор данных (изменяется на месте).
     * @param invert true для обратного преобразования.
     */
    void transform(ComplexVec& v, bool invert = false);

private:
    // Рекурсивная функция управления буферами
    void run_fft(std::span<Complex> data,
                 std::span<Complex> buffer,
                 const ComplexVec& twiddles,
                 size_t current_stride);

    static ComplexVec generate_twiddles(size_t n, bool invert);

    size_t m_max_n;
    ComplexVec m_buffer;
    ComplexVec m_twiddles;  // Таблица для прямого БПФ
    ComplexVec m_itwiddles; // Таблица для обратного БПФ (чтобы не сопрягать в рантайме)
};
