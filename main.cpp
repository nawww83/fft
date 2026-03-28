#include "recursive_fft.hpp"
#include "iterative_fft.hpp"

#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include <ranges>
#include <format> // C++20/23
#include <chrono>
#include <numbers>

////////////////////////////////////////////////////

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

/**
 * Привязывает текущий поток к конкретному логическому ядру.
 * core_id: индекс ядра (0, 1, 2...)
 */
bool pin_thread_to_core(int core_id) {
#if defined(_WIN32) || defined(_WIN64)
    // Windows: SetThreadAffinityMask принимает маску (1 << core_id)
    HANDLE thread = GetCurrentThread();
    DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << core_id);
    if (SetThreadAffinityMask(thread, mask) == 0) {
        return false;
    }
#elif defined(__linux__)
    // Linux: pthread_setaffinity_np
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        return false;
    }
#else
    // Другие ОС (например, macOS) не поддерживают стандартный аффинитет
    return false; 
#endif
    return true;
}

////////////////////////////////////////////////////

// --- Генерация сигнала ---
ComplexVec generate_signal(size_t n) {
    auto view = std::views::iota(0u, n) 

              | std::views::transform([](size_t i) {
                    return Complex{1000.0 * std::sin(i * 0.1), 1000.0 * std::cos(i * 0.2)};
                });
    ComplexVec res; res.reserve(n);
    std::ranges::copy(view, std::back_inserter(res));
    return res;
}

// --- Расчет точности (SNR и L-inf) ---
struct AccuracyMetrics {
    double snr;
    double l_inf;
    std::string status;
};

AccuracyMetrics compute_accuracy(const ComplexVec& original, const ComplexVec& restored) {
    double signal_energy = 0.0, noise_energy = 0.0, l_inf = 0.0, max_amp = 0.0;
    for (auto [orig, rest] : std::views::zip(original, restored)) {
        double a_orig = std::abs(orig);
        double diff = std::abs(orig - rest);
        signal_energy += a_orig * a_orig;
        noise_energy += diff * diff;
        if (a_orig > max_amp) max_amp = a_orig;
        if (diff > l_inf) l_inf = diff;
    }

    double snr = 10.0 * std::log10(signal_energy / (noise_energy + 1e-30));
    double tol = std::numeric_limits<double>::epsilon() * std::max(max_amp, 1.0) * std::log2(original.size() + 1.0);
    
    return {snr, l_inf, (l_inf <= tol ? "OK" : "FAIL")};
}

// --- Бенчмарк ---
template<typename T>
void run_benchmark(std::string_view name, T& fft_processor, size_t n, int iterations) {
    const ComplexVec original = generate_signal(n);
    ComplexVec work = original;

    // Разогрев
    for(int i = 0; i < 5; ++i) { fft_processor.transform(work, false); fft_processor.transform(work, true); }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fft_processor.transform(work, false);
        fft_processor.transform(work, true);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_sec = std::chrono::duration<double>(end - start).count();
    double avg_sec = total_sec / iterations;
    double mflops = (2.0 * 5.0 * n * std::log2(n) / avg_sec) / 1e6;

    // Верификация
    work = original;
    fft_processor.transform(work, false);
    fft_processor.transform(work, true);
    auto acc = compute_accuracy(original, work);

    // Вывод в едином стиле таблицы
    std::cout << std::format("{:<12} | {:<7} | {:>10.4f} | {:>10.1f} | {:>8.1f} | {:>9.1e} | {:>6}\n", 
                             name, n, avg_sec * 1000.0, mflops, acc.snr, acc.l_inf, acc.status);
}

int main() {
    pin_thread_to_core(0);
    try {
        const std::vector<size_t> sizes = {1024, 4096, 16384, 65536};
        
        std::cout << std::format("\n{:=^86}\n", " FFT PERFORMANCE & ACCURACY TEST ");
        std::cout << std::format("{:<12} | {:<7} | {:>10} | {:>10} | {:>8} | {:>9} | {:>6}\n", 
                                 "Algorithm", "N", "Cycle ms", "MFLOPS", "SNR dB", "L-inf", "Stat");
        std::cout << std::format("{:-^86}\n", "");

        for (size_t N : sizes) {
            FFTIterative iterative(N);
            FFTRecursive recursive(N);
            int iters = (N <= 4096) ? 50000 : 5000;

            run_benchmark("Iterative", iterative, N, iters);
            run_benchmark("Recursive", recursive, N, iters);
            std::cout << std::format("{:-^86}\n", "");
        }
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 1; }
    return 0;
}
