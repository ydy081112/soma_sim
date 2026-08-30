#pragma once

#include "soma/common/types.hpp"

#include <cstdint>
#include <limits>

namespace soma {

enum class SpikeKind : std::uint8_t { Data, Bias, SomaDrain };

struct Spike {
    SpikeKind kind = SpikeKind::Data;
    SimTime generated_time = 0;
    SimTime current_time = 0;
    std::uint64_t sequence_id = 0;
    std::uint64_t spike_id = 0;
    std::uint32_t timestep = 0;
    std::size_t source_layer = 0;
    std::uint64_t source_neuron = 0;
    float value = 1.0F;
};

}  // namespace soma
