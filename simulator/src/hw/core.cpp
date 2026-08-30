#include "soma/hw/core.hpp"

#include "soma/hw/synapse.hpp"

#include <limits>
#include <stdexcept>

namespace soma {

Core::Core(const LayerMapping& mapping, const HardwareConfig& hardware, const LayerWeights& weights)
    : mapping_(mapping), hardware_(hardware), weights_(weights),
      soma_(static_cast<std::size_t>(mapping.neurons), mapping.threshold, mapping.leak,
            mapping.reset, mapping.readout) {}

CoreReceiveResult Core::receive(std::uint64_t source_neuron, float value, std::uint32_t timestep,
                                SimTime hw_arrival_time) {
    if (source_neuron >= mapping_.source_neurons) throw std::runtime_error(mapping_.id + ": source neuron 越界");
    const auto input = input_buffer_.reserve(
        hw_arrival_time, hardware_.core.axon_in_hw_latency);
    const auto memory = synapse_sram_.read(
        input.hw_finish_time, hardware_.core.sram_read_hw_latency);

    const auto updates = SynapseEngine::apply(
        weights_, source_neuron, value, mapping_.neurons,
        [&](std::uint64_t destination, float delta) { soma_.accumulate(destination, delta, timestep); });
    if (updates != 0 && hardware_.core.soma_update_hw_latency >
                            std::numeric_limits<SimTime>::max() / updates) {
        throw std::runtime_error("Core hardware compute latency 溢出");
    }
    const SimTime hw_service_latency = hardware_.core.synapse_hw_latency +
                                       updates * hardware_.core.soma_update_hw_latency;
    const auto compute = compute_pipeline_.reserve(
        memory.hw_finish_time, hw_service_latency);
    return CoreReceiveResult{compute.hw_finish_time,
                             compute.hw_finish_time - hw_arrival_time, updates};
}

CoreFireResult Core::drain_one(SimTime hw_arrival_time) {
    const auto reservation = compute_pipeline_.reserve(
        hw_arrival_time, hardware_.core.soma_fire_hw_latency);
    return CoreFireResult{reservation.hw_finish_time,
                          reservation.hw_finish_time - hw_arrival_time,
                          soma_.fire_one()};
}

CoreReceiveResult Core::apply_bias(std::uint32_t timestep, SimTime hw_arrival_time) {
    const auto updates = soma_.apply_bias(weights_.bias, mapping_.output_channels, timestep);
    if (updates != 0 && hardware_.core.soma_update_hw_latency >
                            std::numeric_limits<SimTime>::max() / updates) {
        throw std::runtime_error("Core bias hardware latency 溢出");
    }
    const auto compute = compute_pipeline_.reserve(
        hw_arrival_time, updates * hardware_.core.soma_update_hw_latency);
    return CoreReceiveResult{compute.hw_finish_time,
                             compute.hw_finish_time - hw_arrival_time, updates};
}

}  // namespace soma
