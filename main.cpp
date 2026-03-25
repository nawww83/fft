#include "recursive_fft.hpp"
#include "iterative_fft.hpp"

#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include <ranges>
#include <format> // C++20/23

/**
 * Верификация БПФ с использованием фишек C++23.
 * Адаптивный допуск на основе машинного эпсилона, амплитуды и логарифма N.
 */
static void verify_fft(const ComplexVec &original, const ComplexVec &restored)
{
    if (original.size() != restored.size())
    {
        std::cout << std::format("Ошибка: Размеры векторов не совпадают! ({} vs {})\n",
                                 original.size(), restored.size());
        return;
    }

    const size_t n = original.size();

    // 1. Находим максимальную амплитуду (C++20 ranges)
    // std::views::transform позволяет вычислять abs "на лету" без копирования
    auto abs_view = original | std::views::transform([](auto c)
                                                     { return std::abs(c); });
    double max_amplitude = *std::ranges::max_element(abs_view);

    // Защита от нулевого сигнала
    max_amplitude = std::max(max_amplitude, 1.0);

    // 2. Вычисляем адаптивный порог
    // Ошибка БПФ накапливается пропорционально log2(N)
    constexpr double eps = std::numeric_limits<double>::epsilon();
    const double log2n = std::log2(static_cast<double>(n) + 1.0);
    const double tolerance = eps * max_amplitude * log2n;

    // 3. Сравниваем векторы (C++23 std::views::zip)
    double max_diff = 0.0;
    size_t error_count = 0;

    // zip позволяет идти по двум контейнерам параллельно, возвращая кортеж ссылок
    for (auto [orig, rest] : std::views::zip(original, restored))
    {
        double diff = std::abs(orig - rest);
        if (diff > max_diff)
            max_diff = diff;
        if (diff > tolerance)
            error_count++;
    }

    // 4. Вывод через std::format (C++20/23)
    // {:e} — экспоненциальная форма, {:.15f} — фиксированная точность
    std::cout << std::format(
        "--- Анализ точности (C++23) ---\n"
        "{:<20} {}\n"
        "{:<20} {:.6f}\n"
        "{:<20} {:e}\n"
        "{:<20} {:e}\n"
        "СТАТУС:              {}\n",
        "Размер (N):", n,
        "Max амплитуда:", max_amplitude,
        "Порог (log N):", tolerance,
        "Макс. отклонение:", max_diff,
        (error_count == 0 ? "УСПЕХ" : std::format("ПРОВАЛ ({} ошибок)", error_count)));
}

int main()
{
    {
        size_t N = 2048;
        FFTIterative fft(N);

        // Генерируем сигнал с большой амплитудой для проверки адаптивности
        ComplexVec original(N);
        for (size_t i = 0; i < N; ++i)
        {
            original[i] = {1000.0 * std::sin(i * 0.1), 1000.0 * std::cos(i * 0.2)};
        }

        ComplexVec work = original;
        fft.transform(work, false); // Forward
        fft.transform(work, true);  // Inverse

        verify_fft(original, work);
    }

    {
        try
        {
            size_t N = 2048;

            // 1. Инициализация рекурсивного процессора
            // Выделяет буфер и предвычисляет таблицы один раз
            FFTRecursive fft(N);

            // 2. Генерация тестового сигнала
            ComplexVec original(N);
            for (size_t i = 0; i < N; ++i)
            {
                original[i] = {
                    1000.0 * std::sin(i * 0.1),
                    1000.0 * std::cos(i * 0.2)};
            }

            // 3. Работа с данными (копируем оригинал, чтобы потом сравнить)
            ComplexVec work = original;

            // Прямое преобразование (Forward FFT)
            // Использует рекурсивное разделение и SIMD-бабочки
            fft.transform(work, false);

            // Обратное преобразование (Inverse FFT)
            // Включает автоматическую нормировку на 1/N внутри метода
            fft.transform(work, true);

            // 4. Проверка точности (C++23 версия с адаптивным порогом)
            verify_fft(original, work);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Ошибка: " << e.what() << std::endl;
            return 1;
        }
    }
    return 0;
}
