#pragma once

#include "soma/common/types.hpp"
#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
#include "soma/hw/buffer.hpp"
#include "soma/hw/memory.hpp"
#include "soma/hw/soma.hpp"
#include "soma/runtime/weight_store.hpp"

#include <cstdint>
#include <optional>

namespace soma {

struct CoreReceiveResult {
    SimTime hw_finish_time = 0;
    SimTime hw_compute_latency = 0;
    std::uint64_t synaptic_updates = 0;
};

struct CoreFireResult {
    SimTime hw_finish_time = 0;
    SimTime hw_compute_latency = 0;
    std::optional<FiredNeuron> fired;
};

class Core {
public:
    Core(const LayerMapping& mapping, const HardwareConfig& hardware, const LayerWeights& weights);

    CoreReceiveResult receive(std::uint64_t source_neuron, float value, std::uint32_t timestep,
                              SimTime hw_arrival_time);
    CoreReceiveResult apply_bias(std::uint32_t timestep, SimTime hw_arrival_time);
    CoreFireResult drain_one(SimTime hw_arrival_time);
    bool has_pending_fire() const { return soma_.has_pending(); }
    bool drain_scheduled() const { return drain_scheduled_; }
    void set_drain_scheduled(bool value) { drain_scheduled_ = value; }
    const std::vector<float>& output_scores() const { return soma_.voltage(); }
    bool has_bias() const { return !weights_.bias.empty(); }

private:
    const LayerMapping& mapping_;
    const HardwareConfig& hardware_;
    const LayerWeights& weights_;
    BufferResource input_buffer_;
    BufferResource compute_pipeline_;
    MemoryResource synapse_sram_;
    SomaState soma_;
    bool drain_scheduled_ = false;
};

}  // namespace soma
