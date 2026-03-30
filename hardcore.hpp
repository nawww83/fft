#pragma once

#include <string>      // Для std::string в get_cpu_info
#include <format>

// Кроссплатформенные заголовки
#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#elif defined(__linux__)
    #include <pthread.h>
    #include <sched.h>
    #include <fstream>     // Для чтения /proc/cpuinfo на Linux
#endif

namespace hardcore
{

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

}