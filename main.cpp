#include "recursive_fft.hpp"
#include "iterative_fft.hpp"

#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include <ranges>
#include <format>
#include <chrono>
#include <numbers>
#include <string_view>
#include <limits>

// Кроссплатформенные заголовки для Affinity
#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#elif defined(__linux__)
    #include <pthread.h>
    #include <sched.h>
#endif

struct AccuracyMetrics {
    double snr;
    double l_inf;
    bool is_ok;
};

// Форматтер для вывода Complex через std::format
template <>
struct std::formatter<std::complex<double>, char> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
    auto format(const std::complex<double>& c, format_context& ctx) const {
        return std::format_to(ctx.out(), "{:.2f}+{:.2f}i", c.real(), c.imag());
    }
};

/**
 * @brief Привязка к ядру (Windows/Linux). 
 * На macOS/других просто вернет false.
 */
bool pin_thread_to_core(int core_id) {
#if defined(_WIN32) || defined(_WIN64)
    HANDLE thread = GetCurrentThread();
    DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << core_id);
    return SetThreadAffinityMask(thread, mask) != 0;
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
#else
    (void)core_id; // Подавить warning
    return false; 
#endif
}

ComplexVec generate_signal(size_t n) {
    ComplexVec res;
    res.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        res.emplace_back(std::sin(i * 0.1) * 1000.0, std::cos(i * 0.2) * 1000.0);
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
    double snr = 10.0 * std::log10(signal_energy / (noise_energy + 1e-30));
    double tol = 1e-11 * std::max(max_amp, 1.0) * std::log2(static_cast<double>(original.size()));
    return {snr, l_inf, l_inf <= tol};
}

template<typename T>
void run_benchmark(std::string_view name, T& fft_processor, size_t n, int iterations) {
    const ComplexVec original = generate_signal(n);
    ComplexVec work = original;

    // Разогрев
    for(int i = 0; i < 5; ++i) { 
        fft_processor.transform(work, false); 
        fft_processor.transform(work, true); 
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fft_processor.transform(work, false);
        fft_processor.transform(work, true);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_sec = std::chrono::duration<double>(end - start).count();
    double avg_cycle_ms = (total_sec / iterations) * 1000.0;
    // 2 прохода (FWD+INV) по 5*N*log2(N) операций
    double mflops = (10.0 * n * std::log2(static_cast<double>(n)) / (total_sec / iterations)) / 1e6;

    work = original;
    fft_processor.transform(work, false);
    fft_processor.transform(work, true);
    auto acc = compute_accuracy(original, work);

    std::cout << std::format("{:<12} | {:>8} | {:>10.4f} | {:>10.1f} | {:>8.1f} | {:>9.1e} | {:>6}\n", 
                             name, n, avg_cycle_ms, mflops, acc.snr, acc.l_inf, acc.is_ok ? "OK" : "FAIL");
}

int main() {
    pin_thread_to_core(0);
    try {
        const std::vector<size_t> sizes = {1024, 4096, 16384, 65536};
        
        std::cout << std::format("\n{:^87}\n", "FFT PERFORMANCE & ACCURACY TEST");
        std::cout << std::format("{:-^87}\n", "");
        std::cout << std::format("{:<12} | {:>8} | {:>10} | {:>10} | {:>8} | {:>9} | {:>6}\n", 
                                 "Algorithm", "N", "Cycle (ms)", "MFLOPS", "SNR dB", "L-inf", "Stat");
        std::cout << std::format("{:-^87}\n", "");

        for (size_t N : sizes) {
            FFTIterative iterative(N);
            FFTRecursive recursive(N); // <-- ТЕПЕРЬ ВКЛЮЧЕНО

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
