#pragma once

#include "soma/common/types.hpp"
#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
#include "soma/hw/hardware_resource.hpp"
#include "soma/hw/soma.hpp"
#include "soma/hw/tile.hpp"
#include "soma/runtime/weight_store.hpp"
#include "soma/runtime/incremental_spike_matmul.hpp"
#include "soma/runtime/timestep_spike_attention.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace soma {

struct CoreFiringResult {
    FiredNeuron fired;
    SimTime hw_finish_time = 0;
    std::uint32_t global_core = 0;
    std::uint32_t local_neuron = 0;
};

struct CoreReceiveResult {
    SimTime hw_finish_time = 0;
    SimTime hw_compute_latency = 0;
    SimTime hw_axon_in_service_latency = 0;
    SimTime hw_synapse_service_latency = 0;
    SimTime hw_resource_wait_latency = 0;
    std::uint64_t synaptic_updates = 0;
};

struct CoreNeuronProcessResult {
    SimTime hw_finish_time = 0;
    SimTime hw_compute_latency = 0;
    std::uint64_t mapped_neurons = 0;
    std::uint64_t updated_neurons = 0;
    SimTime hw_soma_service_latency = 0;
    SimTime hw_resource_wait_latency = 0;
    std::uint64_t attention_updates = 0;
    std::uint64_t kv_attention_updates = 0;
    std::uint64_t q_attention_updates = 0;
    SimTime hw_attention_service_latency = 0;
    std::vector<CoreFiringResult> firings;
    std::vector<float> local_state_values;
};

class Core {
public:
    Core(const LayerMapping& mapping, const HardwareConfig& hardware, const LayerWeights& weights,
         PhysicalCoreAddress address, std::uint64_t physical_neuron_begin,
         std::uint64_t physical_neuron_count, std::uint32_t max_connection_delay = 0);

    CoreReceiveResult receive(std::uint64_t source_neuron, float value, std::uint32_t timestep,
                              SimTime hw_arrival_time,
                              const LayerWeights* connection_weights = nullptr,
                              std::uint32_t connection_delay = 0,
                              std::uint32_t destination_axon = 0,
                              const std::string& operand = "",
                              const std::string& operand_layout = "flat_internal");
    CoreNeuronProcessResult process_timestep(std::uint32_t timestep, SimTime hw_arrival_time);
    CoreReceiveResult receive_local_state(std::uint64_t source_neuron, float value,
                                     const LayerWeights& connection_weights);
    const std::vector<float>& output_scores() const;
    const std::vector<std::uint32_t>& output_fire_counts() const;
    const PhysicalCoreAddress& address() const { return address_; }
    std::uint64_t physical_neuron_begin() const { return physical_neuron_begin_; }
    std::uint64_t physical_neuron_count() const { return physical_neuron_count_; }

private:
    const LayerMapping& mapping_;
    const HardwareConfig& hardware_;
    const LayerWeights& weights_;
    PhysicalCoreAddress address_;
    std::uint64_t physical_neuron_begin_ = 0;
    std::uint64_t physical_neuron_count_ = 0;
    HardwareResource compute_pipeline_;
    // multi_valued_state 不需要 LIF/ST-BIF 的 persistent voltage/tracer/fire-count SoA。
    std::unique_ptr<SomaState> soma_;
    std::vector<std::vector<float>> delayed_buffers_;
    std::vector<std::vector<std::uint8_t>> delayed_pending_;
    std::size_t next_buffer_ = 0;
    std::uint32_t processed_timestep_ = 0;
    std::vector<std::uint32_t> last_state_timestep_;
    std::unique_ptr<IncrementalSpikeMatmul> attention_;
    std::unique_ptr<TimestepSpikeAttention> timestep_attention_;
    std::vector<std::unordered_map<std::size_t, std::int8_t>> attention_lhs_buffers_;
    std::vector<std::unordered_map<std::size_t, std::int8_t>> attention_rhs_buffers_;
    std::vector<std::unordered_map<std::size_t, std::int8_t>> attention_third_buffers_;
    std::vector<std::uint8_t> attention_pending_buffers_;
    bool state_started_ = false;
};

}  // namespace soma
