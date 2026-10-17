#include "fft_factory.hpp"
#include "base_fft.hpp"             // Базовый класс
#include "iterative_fft_aos.hpp"    // Конкретный AoS
#include "iterative_fft_soa.hpp"    // Конкретный SoA
#include "recursive_fft_aos.hpp"    //
#include "recursive_fft_soa.hpp"    //
#include "iterative_fft_aos_simd.hpp" // Ручной AVX2
#include "iterative_fft_soa_simd.hpp" // Ручной AVX2
#include "recursive_fft_aos_simd.hpp" // Ручной AVX2
#include "recursive_fft_soa_simd.hpp" // Ручной AVX2
#include <stdexcept>

FFTFactory::FFTFactory(size_t max_n)
    : m_swaps(std::make_shared<const Swaps>(max_n)), m_twiddles(std::make_shared<TwiddleData>(max_n))
{
}

FFTFactory::~FFTFactory() = default;

std::unique_ptr<FFTBase> FFTFactory::create(FFTLayout layout, FFTType type) const
{
    switch (layout)
    {
    case FFTLayout::AoS:
    {
        switch (type)
        {
        case FFTType::Iterative:     return std::make_unique<FFTIterativeAoS>(m_swaps, m_twiddles);
        case FFTType::Recursive:     return std::make_unique<FFTRecursiveAoS>(m_swaps, m_twiddles);
        case FFTType::IterativeSimd: return std::make_unique<FFTIterativeAoS_SIMD>(m_swaps, m_twiddles);
        case FFTType::RecursiveSimd: return std::make_unique<FFTRecursiveAoS_SIMD>(m_swaps, m_twiddles);
        }
        break;
    }
    case FFTLayout::SoA:
    {
        switch (type)
        {
        case FFTType::Iterative:     return std::make_unique<FFTIterativeSoA>(m_swaps, m_twiddles);
        case FFTType::Recursive:     return std::make_unique<FFTRecursiveSoA>(m_swaps, m_twiddles);
        case FFTType::IterativeSimd: return std::make_unique<FFTIterativeSoA_SIMD>(m_swaps, m_twiddles);
        case FFTType::RecursiveSimd: return std::make_unique<FFTRecursiveSoA_SIMD>(m_swaps, m_twiddles);
        }
        break;
    }
    }
    throw std::invalid_argument("Unsupported FFT layout type");
}

std::unique_ptr<FFTBase> FFTFactory::createAoS(FFTType type) const
{
    return create(FFTLayout::AoS, type);
}

std::unique_ptr<FFTBase> FFTFactory::createSoA(FFTType type) const
{
    return create(FFTLayout::SoA, type);
}
