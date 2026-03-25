#include <vector>
#include <cmath>
#include <complex>
#include <numbers>
#include <ranges>
#include <span>
#include <iostream>
#include <format>
#include <algorithm>
#include <random>
#include <bit>

using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;

// 1. ПРЕДВЫЧИСЛЕНИЕ
// Современный вариант.
// ComplexVec generate_all_twiddles(size_t n, bool invert) {
//     double angle_base = (invert ? 2.0 : -2.0) * std::numbers::pi / n;
//     return std::views::iota(0u, n / 2)
//          | std::views::transform([=](size_t k) { return std::polar(1.0, k * angle_base); })
//          | std::ranges::to<ComplexVec>();
// }
// Менее современный вариант.
ComplexVec generate_all_twiddles(size_t n, bool invert) {
    double angle_base = (invert ? 2.0 : -2.0) * std::numbers::pi / n;
    ComplexVec res;
    res.reserve(n / 2);
    
    auto view = std::views::iota(0u, n / 2)
              | std::views::transform([=](size_t k) { return std::polar(1.0, k * angle_base); });
    
    std::ranges::copy(view, std::back_inserter(res));
    return res;
}

// 2. SIMD-ЯДРО
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target_clones("avx2", "avx", "sse4.2", "default")))
#endif
void compute_butterfly_step(std::span<Complex> out, 
                            std::span<const Complex> res_e, 
                            std::span<const Complex> res_o, 
                            std::span<const Complex> twiddles,
                            size_t stride) {
    const size_t half_n = res_e.size();
    for (size_t k = 0; k < half_n; ++k) {
        Complex t = twiddles[k * stride] * res_o[k];
        out[k + 0]      = res_e[k] + t;
        out[k + half_n] = res_e[k] - t;
    }
}

// 3. РЕКУРСИЯ БЕЗ АЛЛОКАЦИЙ
void fft_recursive(std::span<Complex> data, 
                   std::span<Complex> buffer, 
                   const ComplexVec& twiddles, 
                   size_t current_stride) {
    size_t n = data.size();
    if (n <= 1) return;
    size_t half = n / 2;
    // Распределяем четные/нечетные в буфер
    for (size_t i = 0; i < half; ++i) {
        buffer[i + 0]    = data[i * 2 + 0];
        buffer[i + half] = data[i * 2 + 1];
    }
    // Рекурсия: меняем роли data и buffer, чтобы избежать лишних копирований
    // stride удваивается, так как на следующем уровне элементов в 2 раза меньше
    fft_recursive(buffer.subspan(0, half), data.subspan(0, half), twiddles, current_stride * 2);
    fft_recursive(buffer.subspan(half, half), data.subspan(half, half), twiddles, current_stride * 2);

    compute_butterfly_step(data, buffer.subspan(0, half), buffer.subspan(half, half), twiddles, current_stride);
}

// 4. ПУБЛИЧНЫЙ ИНТЕРФЕЙС
/**
 * @brief Вычисляет прямое или обратное БПФ (FFT/IFFT) методом Кули-Тьюки (основание 2).
 * 
 * Особенности реализации:
 * - Рекурсивный алгоритм с разделением на четные/нечетные индексы.
 * - Оптимизация памяти: использует std::span для исключения аллокаций в рекурсии 
 *   (выделяются только два рабочих вектора: для буфера обмена и поворотных коэффициентов).
 * - C++23: генерация коэффициентов через std::views и std::ranges::to.
 * - Нормировка: при invert=true результат делится на N, что обеспечивает 
 *   масштаб данных, идентичный исходному.
 * 
 * @param v Входной вектор (передается по значению для поддержки move-семантики).
 * @param invert Флаг инверсии: false для прямого БПФ, true для обратного.
 * @return ComplexVec Результат преобразования того же размера, что и входной вектор.
 */
ComplexVec fft(ComplexVec v, bool invert = false) {
    size_t n = v.size();
    if (n == 0) return v;
    if (!std::has_single_bit(n)) {
        throw std::invalid_argument("Размер вектора должен быть степенью двойки");
    }
    // Всего две аллокации на весь процесс
    ComplexVec buffer(n);
    ComplexVec twiddles = generate_all_twiddles(n, invert);
    fft_recursive(v, buffer, twiddles, 1);
    if (invert) {
        for (auto& x : v) x /= static_cast<double>(n);
    }
    return v;
}

double get_max_val(const ComplexVec& data) {
    if (data.empty()) return 0.0;
    // Создаем "вид" из модулей и находим максимум
    auto abs_view = data | std::views::transform([](const auto& z) { return std::abs(z); });
    return std::ranges::max(abs_view);
}

void print_complex(const ComplexVec& v, double max_val) {
    if (v.empty()) return;
    size_t n = static_cast<double>(v.size());
    double eps = std::numeric_limits<double>::epsilon();
    double threshold = max_val * eps * (std::log2(n) + 1.0);
    // Лямбда для очистки шума
    auto clean = [threshold](double val) { 
        return std::abs(val) < threshold ? 0.0 : val; 
    };
    for (const auto& z : v) {
        std::cout << std::format("({:.6g}, {:.6g})\n", clean(z.real()), clean(z.imag()));        
    }
}

bool compare_fft_results(const ComplexVec& original, const ComplexVec& reconstructed, double max_val) {
    if (original.size() != reconstructed.size()) return false;
    size_t n = original.size();
    double eps = std::numeric_limits<double>::epsilon();
    // Порог (учитываем накопление ошибки при FFT и IFFT)
    double threshold = max_val * eps * (std::log2(n) + 1.0);
    // Считаем максимальное отклонение между элементами
    // zip объединяет два вектора в пары для сравнения
    auto diff_view = std::views::zip(original, reconstructed) 
                   | std::views::transform([](auto&& pair) {
                       auto [a, b] = pair;
                       return std::abs(a - b);
                   });
    double max_diff = std::ranges::max(diff_view);
    std::cout << std::format("Max difference: {:.6e} (threshold: {:.6e})\n", max_diff, threshold);
    return max_diff <= threshold;
}

/**
 * @brief Генерирует вектор из N случайных комплексных чисел.
 * @param n Размер вектора (желательно степень двойки для FFT).
 * @param min_val Минимальное значение для re и im частей.
 * @param max_val Максимальное значение для re и im частей.
 */
ComplexVec generate_random_complex(size_t n, double min_val = -100.0, double max_val = 100.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min_val, max_val);

    ComplexVec res;
    res.reserve(n); // Важно для производительности

    // Вместо ленивых views используем простой цикл или ranges::generate_n
    // Это надежнее, так как генерация случайных чисел по природе своей не "чистая" функция
    std::generate_n(std::back_inserter(res), n, [&]() {
        return Complex{dist(gen), dist(gen)};
    });

    return res;
}

int main() 
{
    auto data = generate_random_complex(16);
    const auto max_val = get_max_val(data);
    auto reconstructed = fft(fft(data), true);
    if (compare_fft_results(data, reconstructed, max_val)) {
        std::cout << std::format("Success: Reconstruction is accurate!\n");
    } else {
        std::cout << std::format("Warning: Precision loss detected.\n");
    }
    if (reconstructed.size() <= 32)
        print_complex(reconstructed, max_val);
    return 0;
}

