#include "fft_factory.hpp"
#include "base_fft.hpp"
#include "hardcore.hpp"

#include <iomanip>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <thread>
#include <cmath>
#include <chrono>
#include <variant>
#include <string_view>

using RealVec = std::vector<f64>;
using ComplexVec = std::vector<std::complex<f64>>;

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

void run_benchmark(std::string_view name, const std::unique_ptr<FFTBase>& fft_processor, size_t n, int iterations) {
    // --- ПАУЗА ДЛЯ ОХЛАЖДЕНИЯ ЯДРА ---
    // 500-1000 мс достаточно, чтобы сбросить накопленный жар (Thermal Jitter)
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    const ComplexVec original = generate_signal(n);
    
    // Определяем тип данных БПФ
    auto fft_layout = fft_processor->getLayout();

    // Подготавливаем данные в зависимости от типа
    RealVec re(n), im(n);
    ComplexVec work_aos(n);
    std::variant<SoAData, AoSData> data_variant;
    if (fft_layout == FFTLayout::SoA) 
        data_variant = SoAData{re, im};
    else 
        data_variant = AoSData{work_aos};
    std::visit([&](auto& data) {
        data.assign_from(original);
    }, data_variant);

    // Разогрев
    for(int i = 0; i < 500; ++i) { 
        std::visit([&](auto& data) {
            fft_processor->transform(data, false);
            fft_processor->transform(data, true);
        }, data_variant);
    }
    RealVec samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::visit([&](auto& data) {
            fft_processor->transform(data, false);
            fft_processor->transform(data, true);
        }, data_variant);
        auto t2 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    // Расчет точности (один контрольный проход)
    std::visit([&](auto& data) {
        // 1. Загрузка данных
        data.assign_from(original);
        // 2. Расчет
        fft_processor->transform(data, false);
        fft_processor->transform(data, true);
        // 3. Выгрузка результата для проверки точности
        data.extract_to(work_aos);
    }, data_variant);
    auto acc = compute_accuracy(original, work_aos);
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

// Универсальный доступ к данным через compile-time проверку
template <typename T>
auto get_pair(const T& data, size_t k) {
    if constexpr (requires { data.re[k]; }) {
        return std::make_pair(data.re[k], data.im[k]);
    } else {
        return std::make_pair(data.buffer[k].real(), data.buffer[k].imag());
    }
}

template<size_t n>
void test_fft_all_layouts(FFTType fft_type) {
    const double eps = std::numeric_limits<double>::epsilon() * 100;
    FFTFactory factory(n);

    // Вспомогательная функция для жесткой остановки
    auto assert_ok = [](bool condition, const std::string& msg) {
        if (!condition) {
            std::cerr << "\n[КРИТИЧЕСКАЯ ОШИБКА]: " << msg << std::endl;
            throw std::runtime_error("FFT Verification Failed: " + msg);
        }
    };

    // Лямбда для проверки корректности восстановления сигнала
    auto verify = [&](const auto& data, const std::string& label) {
        bool ok = true;
        for (size_t i = 0; i < n; ++i) {        
            auto [val_re, val_im] = get_pair(data, i);
            // Ожидаем 1.0 в центре (n/2), в остальных местах 0.0
            f64 expected_re = (i == n/2) ? 1.0 : 0.0;
            if (std::abs(val_re - expected_re) > eps || std::abs(val_im) > eps) {
                ok = false;
                break;
            }
        }
        std::cout << "Identity " << std::left << std::setw(5) << label 
                << ": [" << (ok ? "УСПЕХ" : "ОШИБКА") << "]\n";
        assert_ok(ok, label + " recovery error");
    };

    // Лямбда для проверки спектра после Forward FFT
    auto verify_spectrum = [&](const auto& data, const std::string& label) {
        bool ok = true;
        for (size_t k = 0; k < n; ++k) {
            f64 expected_re = (k % 2 == 0) ? 1.0 : -1.0;
            auto [val_re, val_im] = get_pair(data, k);
            if (std::abs(val_re - expected_re) > eps || std::abs(val_im) > eps) {
                ok = false;
                // Раскомментируй для отладки конкретной гармоники:
                // std::cout << "Fail at k=" << k << " got (" << val_re << "," << val_im << ")\n";
                break;
            }
        }
        std::cout << "Спектр " << std::left << std::setw(5) << label 
                << ": [" << (ok ? "УСПЕХ" : "ОШИБКА") << "]\n";
        assert_ok(ok, label + " spectrum is wrong");
    };

    std::cout << "--- Запуск комплексного тестирования БПФ, N = " << 
            n << ", type: " << (fft_type == FFTType::Iterative ? "iterative" : "recursive") << 
            " ---\n";

    // 1. ТЕСТ SoA
    std::cout << "SoA\n";
    {
        auto fft = factory.createSoA(fft_type);
        RealVec re(n, 0.0), im(n, 0.0);
        re[n/2] = 1.0;
        SoAData data{re, im};
        fft->transform(data, false); // Forward
        // Сначала проверяем спектр
        verify_spectrum(data, "SoA Spectrum");
        fft->transform(data, true);  // Inverse
        verify(data, "SoA Identity");
    }

    // 2. ТЕСТ AoS
    std::cout << "AoS\n";
    {
        auto fft = factory.createAoS(fft_type);
        ComplexVec buffer(n, {0.0, 0.0});
        buffer[n/2] = {1.0, 0.0};
        AoSData data{buffer};
        fft->transform(data, false);
        // Сначала проверяем спектр
        verify_spectrum(data, "AoS Spectrum");
        fft->transform(data, true);
        verify(data, "AoS Identity");
    }

    // 3. ТЕСТ НА ОШИБКУ (Cross-call protection)
    std::cout << "Cross-call protection\n";
    {
        auto fft = factory.createSoA(fft_type);
        ComplexVec dummy(n);
        AoSData wrong_data{dummy};
        try {
            fft->transform(wrong_data, false);
            std::cout << "Тест Logic: [ОШИБКА] (Исключение не сработало)\n";
        } catch (const std::logic_error& e) {
            std::cout << "Тест Logic: [УСПЕХ] (Защита сработала: " << e.what() << ")\n";
        }
    }
}


int main() {
    // 1. Проверяем доступные ресурсы
    unsigned int total_logical_cores = std::thread::hardware_concurrency();
    
    // Если ядер 8, выберем 4. 
    // Если ядер 4, выберем 2. 
    // Если ядро всего 1, выберем 0.
    int target_core = (total_logical_cores >= 4) ? 4 : (total_logical_cores - 1);

    if (hardcore::pin_thread_to_core(target_core)) {
        std::cout << std::format("Thread pinned to core: {}\n", target_core);
    } else {
        std::cout << "Failed to pin thread, using default scheduler.\n";
    }

    // 2. Инфо о системе
    std::cout << std::format("Hardware: {}\n", hardcore::get_cpu_info());
    std::cout << std::format("Total Logical Cores: {}\n", total_logical_cores);

    test_fft_all_layouts<2>(FFTType::Iterative);
    test_fft_all_layouts<4>(FFTType::Iterative);
    test_fft_all_layouts<8>(FFTType::Iterative);
    test_fft_all_layouts<16>(FFTType::Iterative);
    test_fft_all_layouts<32>(FFTType::Iterative);
    test_fft_all_layouts<64>(FFTType::Iterative);
    test_fft_all_layouts<128>(FFTType::Iterative);
    test_fft_all_layouts<256>(FFTType::Iterative);
    test_fft_all_layouts<512>(FFTType::Iterative);
    test_fft_all_layouts<1024>(FFTType::Iterative);

    test_fft_all_layouts<2>(FFTType::Recursive);
    test_fft_all_layouts<4>(FFTType::Recursive);
    test_fft_all_layouts<8>(FFTType::Recursive);
    test_fft_all_layouts<16>(FFTType::Recursive);
    test_fft_all_layouts<32>(FFTType::Recursive);
    test_fft_all_layouts<64>(FFTType::Recursive);
    test_fft_all_layouts<128>(FFTType::Recursive);
    test_fft_all_layouts<256>(FFTType::Recursive);
    test_fft_all_layouts<512>(FFTType::Recursive);
    test_fft_all_layouts<1024>(FFTType::Recursive);

    // БЕНЧМАРК
    // return 0;
    
    // Печатаем заголовок для Python
    std::cout << "Algorithm,N,Mean,CI95,SNR,L_inf,IsOk\n";

    // Добавляем 524288 и 1048576 для выхода за L3
    const std::vector<size_t> sizes {1024, 4096, 16384, 65536, 262144, 524288, 1048576};
    for (size_t N : sizes) {
        // Между тестами разных размеров N
        target_core = (target_core == 2) ? 4 : 2; // Прыгаем между физическими ядрами
        hardcore::pin_thread_to_core(target_core);

        FFTFactory fft_factory(N);
        auto fft_aos = fft_factory.createAoS(FFTType::Iterative);
        auto fft_soa = fft_factory.createSoA(FFTType::Iterative);
        auto fft_aos_recursive = fft_factory.createAoS(FFTType::Recursive);
        auto fft_soa_recursive = fft_factory.createSoA(FFTType::Recursive);
        int iters = (N <= 16384) ? 50 : 10; // Увеличь если не хватает.

        run_benchmark("Iterative AoS", fft_aos, N, iters);
        run_benchmark("Iterative SoA", fft_soa, N, iters);
        run_benchmark("Recursive AoS", fft_aos_recursive, N, iters);
        run_benchmark("Recursive SoA", fft_soa_recursive, N, iters);
    }
   
    return 0;
}
