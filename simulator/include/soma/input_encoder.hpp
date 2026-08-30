#pragma once

#include "soma/common/types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace soma {

struct InputSpikeRecord {
    SimTime generated_time = 0;
    std::uint32_t timestep = 0;
    std::uint64_t spike_id = 0;
    std::string layer_id;
    std::uint64_t source_neuron = 0;
    float value = 1.0F;
};

struct InputSpikeFile {
    std::vector<InputSpikeRecord> spikes;
    std::optional<int> expected_output;
    std::uint32_t last_timestep = 0;
};

InputSpikeFile load_input_spikes_csv(const std::string& path);

}  // namespace soma
