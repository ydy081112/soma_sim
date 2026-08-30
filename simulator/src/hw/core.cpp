#include "soma/hw/core.hpp"

#include "soma/hw/synapse.hpp"

#include <limits>
#include <stdexcept>

namespace soma {

Core::Core(const LayerMapping& mapping, const HardwareConfig& hardware, const LayerWeights& weights)
    : mapping_(mapping), hardware_(hardware), weights_(weights),
      soma_(static_cast<std::size_t>(mapping.neurons), mapping.threshold, mapping.leak,
            mapping.reset, mapping.readout),
      timestep_buffer_(static_cast<std::size_t>(mapping.neurons), 0.0F),
      timestep_pending_(static_cast<std::size_t>(mapping.neurons), 0) {}

CoreReceiveResult Core::receive(std::uint64_t source_neuron, float value, std::uint32_t timestep,
                                SimTime hw_arrival_time) {
    if (source_neuron >= mapping_.source_neurons) throw std::runtime_error(mapping_.id + ": source neuron 越界");
    // 到达 spike 依次占用 axon-in buffer、SRAM 端口和 compute pipeline。
    const auto input = input_buffer_.reserve(
        hw_arrival_time, hardware_.core.axon_in_hw_latency);
    const auto memory = synapse_sram_.read(
        input.hw_finish_time, hardware_.core.sram_read_hw_latency);

    if (buffered_timestep_ != 0 && buffered_timestep_ != timestep) {
        throw std::logic_error(mapping_.id + ": timestep buffer 尚未同步处理");
    }
    buffered_timestep_ = timestep;

    // SynapseEngine 直接遍历紧凑模板，Data 只累加到下一 timestep 使用的 buffer。
    // 根据 Spatial Pattern / Linear weight, 找到所有受影响 destination neuron
    const auto updates = SynapseEngine::apply(
        weights_, source_neuron, value, mapping_.neurons,
        // 同一 neuron 的多次输入先求和，pending 位区分“没有输入”和“输入和为 0”。
        [&](std::uint64_t destination, float delta) {
            const auto index = static_cast<std::size_t>(destination);
            timestep_buffer_[index] += delta;
            timestep_pending_[index] = 1;
        });

    // Data path 不再包含 soma access/update；synapse lookup 后即完成本次 accumulation。
    const auto compute = compute_pipeline_.reserve(
        memory.hw_finish_time, hardware_.core.synapse_hw_latency);
    return CoreReceiveResult{compute.hw_finish_time,
                             compute.hw_finish_time - hw_arrival_time, updates};
}

CoreNeuronProcessResult Core::process_timestep(std::uint32_t timestep,
                                               SimTime hw_arrival_time) {
    if (buffered_timestep_ != 0 && buffered_timestep_ + 1 != timestep) {
        throw std::logic_error(mapping_.id + ": timestep buffer 与 neuron processing 不连续");
    }

    CoreNeuronProcessResult result;
    result.mapped_neurons = mapping_.neurons;
    SimTime hw_service_latency = 0;
    std::vector<std::pair<FiredNeuron, SimTime>> relative_firings;
    for (std::uint64_t neuron = 0; neuron < mapping_.neurons; ++neuron) {
        const auto index = static_cast<std::size_t>(neuron);
        auto add_latency = [&](SimTime latency) {
            if (latency > std::numeric_limits<SimTime>::max() - hw_service_latency) {
                throw std::runtime_error("Core neuron processing latency 溢出");
            }
            hw_service_latency += latency;
        };
        add_latency(hardware_.core.soma_access_hw_latency);

        const float bias = weights_.bias.empty()
                               ? 0.0F
                               : weights_.bias[index % mapping_.output_channels];
        const auto neuron_result = soma_.process_neuron(
            neuron, timestep_buffer_[index], timestep_pending_[index] != 0, bias);
        if (neuron_result.updated) {
            ++result.updated_neurons;
            add_latency(hardware_.core.soma_update_hw_latency);
        }
        if (neuron_result.fired) {
            add_latency(hardware_.core.soma_fire_hw_latency);
            relative_firings.emplace_back(*neuron_result.fired, hw_service_latency);
        }
        timestep_buffer_[index] = 0.0F;
        timestep_pending_[index] = 0;
    }
    buffered_timestep_ = 0;

    // 一个 Core 的 neuron loop 作为同一 compute resource reservation，内部仍按 neuron id 排序。
    const auto reservation = compute_pipeline_.reserve(hw_arrival_time, hw_service_latency);
    result.hw_finish_time = reservation.hw_finish_time;
    result.hw_compute_latency = reservation.hw_finish_time - hw_arrival_time;
    result.firings.reserve(relative_firings.size());
    for (const auto& firing : relative_firings) {
        result.firings.push_back(
            CoreFiringResult{firing.first, reservation.hw_start_time + firing.second});
    }
    return result;
}

}  // namespace soma
