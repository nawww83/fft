#include "recursive_fft.hpp"
#include "iterative_fft.hpp"
#include "iterative_fft_soa.hpp"
#include "hardcore.hpp"

#include "types.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <string_view> // Для параметров в run_benchmark


struct AccuracyMetrics {
    double snr;
    double l_inf;
    bool is_ok;
};

ComplexVec generate_signal(size_t n) {
    ComplexVec res;
    res.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        res.emplace_back(std::sin(i * 0.13) * 1000.0, std::cos(i * 0.23) * 1000.0);
    }
    return res;
}

AccuracyMetrics compute_accuracy(const ComplexVec& original, const ComplexVec& restored) {
    double signal_energy = 0.0, noise_energy = 0.0, l_inf = 0.0, max_amp = 0.0;
    
    for (size_t i = 0; i < original.size(); ++i) {
        double a_orig = std::abs(original[i]);
        double diff = std::abs(original[i] - restored[i]);
        
        signal_energy += a_orig * a_orig;
        noise_energy += diff * diff;
        max_amp = std::max(max_amp, a_orig);
        l_inf = std::max(l_inf, diff);
    }

    // Если шум экстремально мал, считаем его нулевым
    bool perfect = (noise_energy <= std::numeric_limits<double>::min());
    double snr = perfect ? 999.9 : 10.0 * std::log10(signal_energy / noise_energy);

    // Теоретический предел точности FFT: eps * log2(N) * max_amplitude
    // log2(N) отражает накопление ошибки при "бабочках" (butterflies)
    double eps = std::numeric_limits<double>::epsilon();
    // При -ffast-math точность может упасть, увеличиваем множитель с 1.0 до 16.0-64.0
    double tol = 16. * eps * std::log2(static_cast<double>(original.size())) * std::max(max_amp, 1.0);

    return {snr, l_inf, l_inf <= tol};
}

template<typename T>
void run_benchmark(std::string_view name, T& fft_processor, size_t n, int iterations) {
    const ComplexVec original = generate_signal(n);
    ComplexVec work = original;

    // Разогрев
    for(int i = 0; i < 100; ++i) { 
        fft_processor.transform(work, false); 
        fft_processor.transform(work, true); 
    }

    std::vector<double> samples;
    samples.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        fft_processor.transform(work, false);
        fft_processor.transform(work, true);
        auto t2 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }

    // Расчет точности (один контрольный проход)
    work = original;
    fft_processor.transform(work, false);
    fft_processor.transform(work, true);
    auto acc = compute_accuracy(original, work);

    // Статистика времени
    double sum = 0, sq_sum = 0;
    for(double s : samples) { sum += s; sq_sum += s*s; }
    double avg = sum / iterations;
    double std_dev = std::sqrt(std::abs(sq_sum/iterations - avg*avg));
    double ci95 = 1.96 * (std_dev / std::sqrt(static_cast<double>(iterations)));

    // CSV: Algo, N, Mean, CI95, SNR, L_inf, IsOk
    std::cout << std::format("{},{},{:.6f},{:.6f},{:.2f},{:.2e},{}\n", 
                             name, n, avg, ci95, acc.snr, acc.l_inf, acc.is_ok ? 1 : 0);
}

void test_iterative_fft_soa()
{
    // 1. Подготовка данных
    // Используем размер 16 для наглядности вывода
    size_t n = 16;
    ComplexVec work(n, Complex(0.0, 0.0));
    // Установим единицу в 8-й элемент (импульс со сдвигом)
    work[8] = {1.0, 0.0};

    // 2. Инициализация процессора FFT
    // Резервируем таблицы под максимальный размер
    FFTIterativeSoA fft_processor(n);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "--- Исходный вектор (импульс в work[8]) ---\n";
    for (size_t i = 0; i < n; ++i) {
        std::cout << i << ": (" << work[i].real() << ", " << work[i].imag() << ")\n";
    }
    // 3. Прямое преобразование (Forward FFT)
    fft_processor.transform(work, false);
    std::cout << "\n--- После прямого FFT (Спектр) ---\n";
    for (const auto& c : work) {
        // Убираем микроскопические значения для чистоты вывода
        double re = std::abs(c.real()) < 1e-15 ? 0.0 : c.real();
        double im = std::abs(c.imag()) < 1e-15 ? 0.0 : c.imag();
        std::cout << "(" << re << ", " << im << "); ";
    }
    std::cout << "\n";
    // 4. Обратное преобразование (Inverse FFT)
    fft_processor.transform(work, true);

    std::cout << "\n--- После обратного FFT (Восстановление) ---\n";
    bool success = true;
    for (size_t i = 0; i < n; ++i) {
        double re = std::abs(work[i].real()) < 1e-15 ? 0.0 : work[i].real();
        double im = std::abs(work[i].imag()) < 1e-15 ? 0.0 : work[i].imag();
        
        std::cout << i << ": (" << re << ", " << im << ")\n";
        
        // Проверка: должен вернуться исходный импульс в work[8]
        if (i == 8 && std::abs(re - 1.0) > 1e-9) success = false;
        if (i != 8 && (std::abs(re) > 1e-9 || std::abs(im) > 1e-9)) success = false;
    }
    std::cout << "\nРезультат: " << (success ? "УСПЕХ (Данные восстановлены)" : "ОШИБКА") << std::endl;
}

int main() {
    hardcore::pin_thread_to_core(0);
    std::cout << std::format("\nHardware: {}\n", hardcore::get_cpu_info());
    
    // Печатаем заголовок для Python
    std::cout << "Algorithm,N,Mean,CI95,SNR,L_inf,IsOk\n";

    // Добавляем 524288 и 1048576 для выхода за L3
    const std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144, 524288, 1048576};
    for (size_t N : sizes) {
        FFTIterative iterative(N);
        FFTRecursive recursive(N);
        FFTIterativeSoA iterative_soa(N);
        // Для больших N уменьшаем итерации до 200, чтобы тест не шел вечно
        int iters = (N <= 16384) ? 5000 : 200;

        run_benchmark("Iterative", iterative, N, iters);
        run_benchmark("Recursive", recursive, N, iters);
        run_benchmark("Iterative SOA", iterative_soa, N, iters);
    }

    // test_iterative_fft_soa();
   
    return 0;
}
