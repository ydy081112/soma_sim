#pragma once

#include "soma/common/types.hpp"
#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
#include "soma/hw/hardware_resource.hpp"
#include "soma/hw/memory.hpp"
#include "soma/hw/soma.hpp"
#include "soma/runtime/weight_store.hpp"

#include <cstdint>
#include <vector>

namespace soma {

struct CoreFiringResult {
    FiredNeuron fired;
    SimTime hw_finish_time = 0;
};

struct CoreReceiveResult {
    SimTime hw_finish_time = 0;
    SimTime hw_compute_latency = 0;
    std::uint64_t synaptic_updates = 0;
};

struct CoreNeuronProcessResult {
    SimTime hw_finish_time = 0;
    SimTime hw_compute_latency = 0;
    std::uint64_t mapped_neurons = 0;
    std::uint64_t updated_neurons = 0;
    std::vector<CoreFiringResult> firings;
};

class Core {
public:
    Core(const LayerMapping& mapping, const HardwareConfig& hardware, const LayerWeights& weights);

    CoreReceiveResult receive(std::uint64_t source_neuron, float value, std::uint32_t timestep,
                              SimTime hw_arrival_time);
    CoreNeuronProcessResult process_timestep(std::uint32_t timestep, SimTime hw_arrival_time);
    const std::vector<float>& output_scores() const { return soma_.voltage(); }

private:
    const LayerMapping& mapping_;
    const HardwareConfig& hardware_;
    const LayerWeights& weights_;
    HardwareResource input_buffer_;
    HardwareResource compute_pipeline_;
    MemoryResource synapse_sram_;
    SomaState soma_;
    std::vector<float> timestep_buffer_;
    std::vector<std::uint8_t> timestep_pending_;
    std::uint32_t buffered_timestep_ = 0;
};

}  // namespace soma
