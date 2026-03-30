#include "recursive_fft.hpp"
#include "iterative_fft.hpp"

#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include <format>
#include <chrono>
#include <string>      // Для std::string в get_cpu_info
#include <string_view> // Для параметров в run_benchmark
#include <fstream>     // Для чтения /proc/cpuinfo на Linux

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

std::string get_cpu_info() {
    std::string model = "Unknown CPU";
    double freq_mhz = 0.0;

#if defined(_WIN32) || defined(_WIN64)
    HKEY hKey;
    char buffer[256];
    DWORD buffer_size = sizeof(buffer);
    DWORD freq_val = 0;
    DWORD freq_size = sizeof(freq_val);

    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        // Модель
        if (RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL, (LPBYTE)buffer, &buffer_size) == ERROR_SUCCESS) {
            model = buffer;
        }
        // Частота (МГц)
        if (RegQueryValueExA(hKey, "~MHz", NULL, NULL, (LPBYTE)&freq_val, &freq_size) == ERROR_SUCCESS) {
            freq_mhz = static_cast<double>(freq_val);
        }
        RegCloseKey(hKey);
    }
#elif defined(__linux__)
    // Модель из /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.starts_with("model name")) {
            model = line.substr(line.find(": ") + 2);
            break;
        }
    }
    // Максимальная частота из sysfs (в кГц)
    std::ifstream freq_file("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (freq_file.is_open()) {
        long khz;
        freq_file >> khz;
        freq_mhz = khz / 1000.0;
    }
#endif
    return std::format("{} @ {:.2f} GHz", model, freq_mhz / 1000.0);
}

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

    // Безопасный расчет SNR для -ffast-math
    double snr;
    if (noise_energy > std::numeric_limits<double>::min()) {
        snr = 10.0 * std::log10(signal_energy / noise_energy);
    } else {
        // Вместо inf используем очень большое число (например, 999.0 dB)
        // Это наглядно в таблице и не сломает логику форматтера
        snr = 999.0; 
    }

    // Теоретический предел точности FFT: eps * log2(N) * max_amplitude
    // log2(N) отражает накопление ошибки при "бабочках" (butterflies)
    double eps = std::numeric_limits<double>::epsilon();
    double tol = eps * std::log2(static_cast<double>(original.size())) * std::max(max_amp, 1.0);

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
        std::cout << std::format("\nHardware: {}\n", get_cpu_info());
        const std::vector<size_t> sizes = {1024, 4096, 16384, 65536};
        
        std::cout << std::format("{:^87}\n", "FFT PERFORMANCE & ACCURACY TEST");

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
