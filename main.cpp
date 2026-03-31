#include "recursive_fft.hpp"
#include "iterative_fft.hpp"
#include "hardcore.hpp"

#include <iostream>
#include <vector>
#include <complex>
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

    // 1. Разогрев
    for(int i = 0; i < 50; ++i) { 
        fft_processor.transform(work, false); 
        fft_processor.transform(work, true); 
    }

    std::vector<double> samples;
    samples.reserve(iterations);

    // 2. Сбор индивидуальных замеров
    for (int i = 0; i < iterations; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        fft_processor.transform(work, false);
        fft_processor.transform(work, true);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        samples.push_back(std::chrono::duration<double>(t2 - t1).count());
    }

    // 3. Статистика
    std::sort(samples.begin(), samples.end());
    
    [[maybe_unused]] double min_sec = samples.front();
    double median_sec = samples[iterations / 2];
    
    double sum = 0, sq_sum = 0;
    for(double s : samples) { sum += s; sq_sum += s*s; }
    double avg_sec = sum / iterations;
    double std_dev_perc = std::sqrt(std::abs(sq_sum/iterations - avg_sec*avg_sec)) / avg_sec * 100.0;

    // 4. Расчет MFLOPS на основе МЕДИАНЫ (самый честный показатель)
    double logN = std::log2(static_cast<double>(n));
    double total_ops = (10.0 * n * logN) + static_cast<double>(n);
    double mflops = (total_ops / median_sec) / 1e6;

    // Точность (один проход)
    work = original;
    fft_processor.transform(work, false);
    fft_processor.transform(work, true);
    auto acc = compute_accuracy(original, work);

    // Вывод: добавили Jitter (std_dev) в процентах
    // Выводим Jitter с символом %, SNR и статус
    std::cout << std::format("{:<12} | {:>8} | {:>10.4f} | {:>10.1f} | {:>7.1f}% | {:>8.1f} | {:>6}\n", 
                         name, n, median_sec * 1000.0, mflops, std_dev_perc, acc.snr, acc.is_ok ? "OK" : "FAIL");
}


int main() {
    hardcore::pin_thread_to_core(0);
    try {
        std::cout << std::format("\nHardware: {}\n", hardcore::get_cpu_info());
        const std::vector<size_t> sizes = {1024, 4096, 16384, 65536};
        
        std::cout << std::format("{:^87}\n", "FFT PERFORMANCE & ACCURACY TEST");

        std::cout << std::format("{:-^87}\n", "");
        std::cout << std::format("{:<12} | {:>8} | {:>10} | {:>10} | {:>8} | {:>8} | {:>6}\n", 
                         "Algorithm", "N", "Cycle (ms)", "MFLOPS", "Jitter", "SNR dB", "Stat");
        std::cout << std::format("{:-^87}\n", "");

        for (size_t N : sizes) {
            FFTIterative iterative(N);
            FFTRecursive recursive(N);

            int iters = (N <= 4096) ? 5000 : 500;

            run_benchmark("Iterative", iterative, N, iters);
            run_benchmark("Recursive", recursive, N, iters);
            std::cout << std::format("{:-^87}\n", "");
        }
    } catch (const std::exception& e) { 
        std::cerr << "Error: " << e.what() << "\n"; 
        return 1; 
    }
    return 0;
}
