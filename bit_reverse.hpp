#pragma once

#include <bit>
#ifdef _MSC_VER
    #include <intrin.h>
#endif

// Универсальный разворот бит
inline size_t fast_bit_reverse(size_t i, int log2n) {
#if defined(__cpp_lib_bit_reverse) && !defined(_MSC_VER)
    // GCC / Clang с поддержкой C++23
    return std::bit_reverse(i) >> (sizeof(size_t) * 8 - log2n);
#elif defined(_MSC_VER)
    // Microsoft Visual C++ (любая версия)
    #if defined(_M_X64)
        uint64_t v = static_cast<uint64_t>(i);
        v = ((v & 0x5555555555555555ull) << 1) | ((v & 0xAAAAAAAAAAAAAAAAull) >> 1);
        v = ((v & 0x3333333333333333ull) << 2) | ((v & 0xCCCCCCCCCCCCCCCCull) >> 2);
        v = ((v & 0x0F0F0F0F0F0F0F0Full) << 4) | ((v & 0xF0F0F0F0F0F0F0F0ull) >> 4);
        return static_cast<size_t>(_byteswap_uint64(v) >> (64 - log2n));
    #else
        uint32_t v = static_cast<uint32_t>(i);
        v = ((v & 0x55555555) << 1) | ((v & 0xAAAAAAAA) >> 1);
        v = ((v & 0x33333333) << 2) | ((v & 0xCCCCCCCC) >> 2);
        v = ((v & 0x0F0F0F0F) << 4) | ((v & 0xF0F0F0F0) >> 4);
        return static_cast<size_t>(_byteswap_ulong(v) >> (32 - log2n));
    #endif
#else
    // Фолбэк для старых стандартов GCC/Clang (C++20 и ниже)
    size_t res = 0;
    for (int b = 0; b < log2n; ++b) {
        if (i & (size_t(1) << b)) res |= (size_t(1) << (log2n - 1 - b));
    }
    return res;
#endif
}