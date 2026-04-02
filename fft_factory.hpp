#pragma once

#include "types.hpp"
#include <memory>

class FFTBase;

class FFTFactory {
public:
    explicit FFTFactory(size_t max_n);
    ~FFTFactory();

    std::unique_ptr<FFTBase> create(FFTLayout layout, FFTType type) const;

    std::unique_ptr<FFTBase> createAoS(FFTType type) const;
    std::unique_ptr<FFTBase> createSoA(FFTType type) const;

private:
    SwapsSP m_swaps;
    TwiddleSP m_twiddles;
};
