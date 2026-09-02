#include "soma/hw/core.hpp"

#include "soma/hw/synapse.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace soma {

Core::Core(const LayerMapping& mapping, const HardwareConfig& hardware, const LayerWeights& weights,
           PhysicalCoreAddress address, std::uint64_t physical_neuron_begin,
           std::uint64_t physical_neuron_count, std::uint32_t max_connection_delay)
    : mapping_(mapping), hardware_(hardware), weights_(weights), address_(address),
      physical_neuron_begin_(physical_neuron_begin),
      physical_neuron_count_(physical_neuron_count),
      soma_(static_cast<std::size_t>(physical_neuron_count), mapping.threshold, mapping.leak,
            mapping.reset, mapping.readout, mapping.membrane_quantization_step,
            mapping.threshold_comparison),
      delayed_buffers_(static_cast<std::size_t>(max_connection_delay) + 1,
                       std::vector<float>(static_cast<std::size_t>(physical_neuron_count), 0.0F)),
      delayed_pending_(static_cast<std::size_t>(max_connection_delay) + 1,
                       std::vector<std::uint8_t>(static_cast<std::size_t>(physical_neuron_count), 0)) {
    if (physical_neuron_count == 0 ||
        physical_neuron_begin + physical_neuron_count > mapping.neurons ||
        physical_neuron_count > hardware.core.max_neurons) {
        throw std::runtime_error(mapping.id + ": physical Core neuron range 不合法");
    }
}

CoreReceiveResult Core::receive(std::uint64_t source_neuron, float value, std::uint32_t timestep,
                                SimTime hw_arrival_time,
                                const LayerWeights* connection_weights,
                                std::uint32_t connection_delay) {
    const auto& active_weights = connection_weights == nullptr ? weights_ : *connection_weights;
    if (source_neuron >= active_weights.source_neurons && active_weights.source_neurons != 0)
        throw std::runtime_error(mapping_.id + ": source neuron 越界");
    // 独立 Core 单测可省略初始空 neuron phase；完整 Simulator 始终先执行该 phase。
    if (processed_timestep_ == 0 && timestep == 1) processed_timestep_ = 1;
    if (timestep != processed_timestep_) throw std::logic_error(mapping_.id + ": Data 必须在同 timestep neuron phase 后到达");
    if (connection_delay >= delayed_buffers_.size()) throw std::runtime_error(mapping_.id + ": connection delay 超出配置");
    const auto buffer = (next_buffer_ + connection_delay) % delayed_buffers_.size();

    // SynapseEngine 直接遍历紧凑模板，Data 只累加到下一 timestep 使用的 buffer。
    // 根据 Spatial Pattern / Linear weight, 找到所有受影响 destination neuron
    const auto physical_end = physical_neuron_begin_ + physical_neuron_count_;
    const auto updates = SynapseEngine::apply_to_physical_range(
        active_weights, source_neuron, value, mapping_, physical_neuron_begin_, physical_end,
        // 同一 neuron 的多次输入先求和，pending 位区分“没有输入”和“输入和为 0”。
        [&](std::uint64_t destination, float delta) {
            const auto physical = mapping_.physical_neuron_index(destination);
            const auto index = static_cast<std::size_t>(physical - physical_neuron_begin_);
            delayed_buffers_[buffer][index] += delta;
            delayed_pending_[buffer][index] = 1;
        });

    // Spatial Pattern 与 source-major Dense 保持各自的可配置 synapse service cost。
    const auto hardware_type = connection_weights == nullptr && active_weights.op == LayerOp::Linear
                                   ? ConnectionType::Dense
                                   : active_weights.hardware_type;
    const auto synapse_hw_latency = hardware_type == ConnectionType::Identity
                                        ? hardware_.core.identity_synapse_hw_latency
                                    : hardware_type == ConnectionType::Dense
                                        ? hardware_.core.dense_synapse_hw_latency
                                        : hardware_.core.spatial_synapse_hw_latency;
    if (updates > (std::numeric_limits<SimTime>::max() /
                   std::max<SimTime>(synapse_hw_latency, 1))) {
        throw std::runtime_error(mapping_.id + ": synapse service latency 溢出");
    }
    const auto synapse_service = updates * synapse_hw_latency;
    const auto packet_service = hardware_.core.axon_in_hw_latency +
                                hardware_.core.sram_read_hw_latency + synapse_service;
    // SANA-FE 的 destination Core 按 packet 串行处理 axon-in 与全部 local synapses。
    const auto compute = compute_pipeline_.reserve(hw_arrival_time, packet_service);
    return CoreReceiveResult{compute.hw_finish_time,
                             compute.hw_finish_time - hw_arrival_time,
                             hardware_.core.axon_in_hw_latency,
                             synapse_service,
                             compute.hw_wait_latency,
                             updates};
}

CoreNeuronProcessResult Core::process_timestep(std::uint32_t timestep,
                                               SimTime hw_arrival_time) {
    if (timestep != processed_timestep_ + 1) throw std::logic_error(mapping_.id + ": neuron timestep 不连续");
    auto& timestep_buffer = delayed_buffers_[next_buffer_];
    auto& timestep_pending = delayed_pending_[next_buffer_];

    CoreNeuronProcessResult result;
    result.mapped_neurons = physical_neuron_count_;
    SimTime hw_service_latency = 0;
    std::vector<std::pair<FiredNeuron, SimTime>> relative_firings;
    for (std::uint64_t local_neuron = 0; local_neuron < physical_neuron_count_; ++local_neuron) {
        const auto index = static_cast<std::size_t>(local_neuron);
        const auto physical_neuron = physical_neuron_begin_ + local_neuron;
        const auto neuron = mapping_.logical_neuron_index(physical_neuron);
        auto add_latency = [&](SimTime latency) {
            if (latency > std::numeric_limits<SimTime>::max() - hw_service_latency) {
                throw std::runtime_error("Core neuron processing latency 溢出");
            }
            hw_service_latency += latency;
        };
        add_latency(hardware_.core.soma_access_hw_latency);

        const float bias = weights_.bias.empty()
                               ? 0.0F
                               : weights_.bias[static_cast<std::size_t>(
                                     neuron % mapping_.output_channels)];
        const auto neuron_result = soma_.process_neuron(
            local_neuron, timestep_buffer[index], timestep_pending[index] != 0, bias);
        if (neuron_result.updated) {
            ++result.updated_neurons;
            add_latency(hardware_.core.soma_update_hw_latency);
        }
        if (neuron_result.fired) {
            add_latency(hardware_.core.soma_fire_hw_latency);
            relative_firings.emplace_back(
                FiredNeuron{neuron, neuron_result.fired->value}, hw_service_latency);
        }
        timestep_buffer[index] = 0.0F;
        timestep_pending[index] = 0;
    }
    next_buffer_ = (next_buffer_ + 1) % delayed_buffers_.size();
    processed_timestep_ = timestep;

    // 一个 Core 的 neuron loop 作为同一 compute resource reservation，内部仍按 neuron id 排序。
    const auto reservation = compute_pipeline_.reserve(hw_arrival_time, hw_service_latency);
    result.hw_finish_time = reservation.hw_finish_time;
    result.hw_compute_latency = reservation.hw_finish_time - hw_arrival_time;
    result.hw_soma_service_latency = hw_service_latency;
    result.hw_resource_wait_latency = reservation.hw_wait_latency;
    result.firings.reserve(relative_firings.size());
    for (const auto& firing : relative_firings) {
        result.firings.push_back(
            CoreFiringResult{firing.first, reservation.hw_start_time + firing.second});
    }
    return result;
}

}  // namespace soma
