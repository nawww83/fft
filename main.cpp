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

// --- Функция генерации сигнала (совместимая с GCC 13.1) ---
ComplexVec generate_signal(size_t n) {
    auto view = std::views::iota(0u, n) 
              | std::views::transform([](size_t i) {
                    return Complex{1000.0 * std::sin(i * 0.1), 1000.0 * std::cos(i * 0.2)};
                });
    
    ComplexVec res;
    res.reserve(n); // Важно для производительности
    
    // Используем алгоритм из ranges, он умеет работать с sentinel (view.end())
    std::ranges::copy(view, std::back_inserter(res));
    
    return res;
}

// --- Ваша функция верификации ---
static void verify_fft(const ComplexVec &original, const ComplexVec &restored) {
    const size_t n = original.size();
    auto abs_view = original | std::views::transform([](auto c) { return std::abs(c); });
    double max_amplitude = std::max(*std::ranges::max_element(abs_view), 1.0);

    constexpr double eps = std::numeric_limits<double>::epsilon();
    const double tolerance = eps * max_amplitude * std::log2(static_cast<double>(n) + 1.0);

    double max_diff = 0.0;
    size_t error_count = 0;

    for (auto [orig, rest] : std::views::zip(original, restored)) {
        double diff = std::abs(orig - rest);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) error_count++;
    }

    std::cout << std::format("   [Verify] Max Diff: {:e} | Status: {}\n", 
                             max_diff, (error_count == 0 ? "OK" : "FAIL"));
}

// --- Функция бенчмарка ---
template<typename T>
void run_benchmark(std::string_view name, T& fft_processor, size_t n, int iterations = 500) {
    const ComplexVec original = generate_signal(n);
    ComplexVec work = original;

    // 1. Разогрев (Warm-up)
    for(int i = 0; i < 3; ++i) {
        fft_processor.transform(work, false);
        fft_processor.transform(work, true);
    }

    // 2. Замер только прямого преобразования (Forward)
    // Чтобы данные не "портились", перед каждой итерацией 
    // в идеале нужно восстанавливать вектор, но для замера скорости 
    // процессору всё равно, какие там числа (NaN не в счет).
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fft_processor.transform(work, false);
        // Если вы хотите мерить и обратное, делайте так:
        // fft_processor.transform(work, true); 
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    // Выводим среднее время одного прохода (Forward)
    std::cout << std::format("{:<12} | N: {:<7} | Avg: {:>8.4f} ms\n", 
                             name, n, total_ms / iterations);
    
    // 3. ОДНОКРАТНАЯ верификация в конце на свежих данных
    work = original;
    fft_processor.transform(work, false);
    fft_processor.transform(work, true);
    verify_fft(original, work); 
}


int main() {
    if (pin_thread_to_core(0)) {
        std::cout << "Thread pinned to core 0\n";
    } else {
        std::cerr << "Failed to pin thread\n";
    }
    try {
        const std::vector<size_t> sizes = {1024, 4096, 16384, 65536};
        
        std::cout << std::format("{:=^60}\n", " FFT PERFORMANCE TEST ");
        
        for (size_t N : sizes) {
            FFTIterative iterative(N);
            FFTRecursive recursive(N);

            run_benchmark("Iterative", iterative, N, 500);
            run_benchmark("Recursive", recursive, N, 500);
            std::cout << std::format("{:-^60}\n", "");
        }
    } catch (const std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }
    return 0;
}
