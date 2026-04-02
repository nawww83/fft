#pragma once
#include "types.hpp"
#include <span>
#include <stdexcept>

class FFTBase {
public:
    explicit FFTBase(SwapsSP swaps, TwiddleSP twiddles) 
    : m_swaps(std::move(swaps)) 
    , m_twiddles(std::move(twiddles))
    {
        if (!m_swaps) throw std::invalid_argument("Swaps table cannot be null");
         if (!m_twiddles) throw std::invalid_argument("Twiddles cannot be null");
    }

    virtual ~FFTBase() = default;

    virtual FFTLayout getType() const = 0;

    virtual void transform([[maybe_unused]] AoSData& data, [[maybe_unused]] bool invert) {
        throw std::logic_error("AoS transform not supported by this engine");
    }

    virtual void transform([[maybe_unused]] SoAData& data, [[maybe_unused]] bool invert) {
        throw std::logic_error("SoA transform not supported by this engine");
    }

protected:
    SwapsSP m_swaps;
    TwiddleSP m_twiddles;
};
