#include "soma/hw/core.hpp"

#include "soma/hw/synapse.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace soma {
namespace {

std::vector<float> local_values(const std::vector<float>& values, const LayerMapping& mapping,
                                std::uint64_t begin, std::uint64_t count) {
    if (values.empty()) return {};
    std::vector<float> result(static_cast<std::size_t>(count));
    for (std::uint64_t local = 0; local < count; ++local) {
        const auto logical = mapping.logical_neuron_index(begin + local);
        const auto index = values.size() == 1 ? 0
                         : values.size() == mapping.neurons ? static_cast<std::size_t>(logical)
                         : static_cast<std::size_t>(logical % mapping.output_channels);
        result[static_cast<std::size_t>(local)] = values.at(index);
    }
    return result;
}

}  // namespace

Core::Core(const LayerMapping& mapping, const HardwareConfig& hardware, const LayerWeights& weights,
           PhysicalCoreAddress address, std::uint64_t physical_neuron_begin,
           std::uint64_t physical_neuron_count, std::uint32_t max_connection_delay)
    : mapping_(mapping), hardware_(hardware), weights_(weights), address_(address),
      physical_neuron_begin_(physical_neuron_begin),
      physical_neuron_count_(physical_neuron_count),
      delayed_buffers_(static_cast<std::size_t>(max_connection_delay) + 1,
                       std::vector<float>(static_cast<std::size_t>(physical_neuron_count), 0.0F)),
      delayed_pending_(static_cast<std::size_t>(max_connection_delay) + 1,
                       std::vector<std::uint8_t>(static_cast<std::size_t>(physical_neuron_count), 0)) {
    if (physical_neuron_count == 0 ||
        physical_neuron_begin + physical_neuron_count > mapping.neurons ||
        physical_neuron_count > hardware.core.max_neurons) {
        throw std::runtime_error(mapping.id + ": physical Core neuron range 不合法");
    }
    if (mapping.neuron_model != "multi_valued_state") {
        last_state_timestep_.assign(static_cast<std::size_t>(physical_neuron_count), 0);
        soma_ = std::make_unique<SomaState>(
            static_cast<std::size_t>(physical_neuron_count), mapping.threshold, mapping.leak,
            mapping.reset, mapping.readout, mapping.membrane_quantization_step,
            mapping.threshold_comparison, hardware.core.threshold_compare_mode,
            local_values(weights.threshold, mapping, physical_neuron_begin, physical_neuron_count),
            hardware.core.membrane_min, hardware.core.membrane_max,
            hardware.core.fire_on_positive_saturation, mapping.neuron_model,
            mapping.tracer_min, mapping.tracer_max,
            local_values(weights.initial_membrane, mapping, physical_neuron_begin,
                         physical_neuron_count));
    }
    if (mapping.operator_type == "incremental_spike_matmul") {
        const auto kind = mapping.attention_kind == "qk"
                              ? IncrementalSpikeMatmul::Kind::Qk
                              : IncrementalSpikeMatmul::Kind::Av;
        attention_ = std::make_unique<IncrementalSpikeMatmul>(
            kind, mapping.attention_heads, mapping.attention_rows,
            mapping.attention_reduction, mapping.attention_columns,
            physical_neuron_begin, physical_neuron_count,
            mapping.attention_output_layout == "row_head_column");
        const auto lhs_size = static_cast<std::size_t>(mapping.attention_heads) *
                              mapping.attention_rows * mapping.attention_reduction;
        const auto rhs_size = static_cast<std::size_t>(mapping.attention_heads) *
                              mapping.attention_columns * mapping.attention_reduction;
        attention_lhs_buffers_.resize(delayed_buffers_.size());
        attention_rhs_buffers_.resize(delayed_buffers_.size());
        static_cast<void>(lhs_size);
        static_cast<void>(rhs_size);
        attention_pending_buffers_.assign(delayed_buffers_.size(), 0);
    }
    if (mapping.operator_type == "timestep_spike_attention") {
        timestep_attention_ = std::make_unique<TimestepSpikeAttention>(
            mapping.attention_heads, mapping.attention_rows, mapping.attention_reduction,
            physical_neuron_begin, physical_neuron_count, mapping.attention_scale);
        attention_lhs_buffers_.resize(delayed_buffers_.size());
        attention_rhs_buffers_.resize(delayed_buffers_.size());
        attention_third_buffers_.resize(delayed_buffers_.size());
        attention_pending_buffers_.assign(delayed_buffers_.size(), 0);
    }
}

CoreReceiveResult Core::receive(std::uint64_t source_neuron, float value, std::uint32_t timestep,
                                SimTime hw_arrival_time,
                                const LayerWeights* connection_weights,
                                std::uint32_t connection_delay,
                                std::uint32_t destination_axon,
                                const std::string& operand,
                                const std::string& operand_layout) {
    const auto& active_weights = connection_weights == nullptr ? weights_ : *connection_weights;
    if (source_neuron >= active_weights.source_neurons && active_weights.source_neurons != 0)
        throw std::runtime_error(mapping_.id + ": source neuron 越界");
    // 独立 Core 单测可省略初始空 neuron phase；完整 Simulator 始终先执行该 phase。
    if (processed_timestep_ == 0 && timestep == 1) processed_timestep_ = 1;
    if (timestep != processed_timestep_) throw std::logic_error(mapping_.id + ": Data 必须在同 timestep neuron phase 后到达");
    if (connection_delay >= delayed_buffers_.size()) throw std::runtime_error(mapping_.id + ": connection delay 超出配置");
    const auto buffer = (next_buffer_ + connection_delay) % delayed_buffers_.size();
    state_started_ = true;

    if (active_weights.connection_type == ConnectionType::AttentionOperand) {
        const bool incremental = attention_ != nullptr && (operand == "lhs" || operand == "rhs");
        const bool timestep_attention = timestep_attention_ != nullptr &&
                                        (operand == "q" || operand == "k" || operand == "v");
        if (!incremental && !timestep_attention)
            throw std::runtime_error(mapping_.id + ": attention operand 配置无效");
        auto& target = operand == "lhs" || operand == "q" ? attention_lhs_buffers_[buffer]
                     : operand == "rhs" || operand == "k" ? attention_rhs_buffers_[buffer]
                                                               : attention_third_buffers_[buffer];
        std::size_t index = static_cast<std::size_t>(source_neuron);
        if (operand_layout == "row_head_reduction") {
            const auto per_row = static_cast<std::uint64_t>(mapping_.attention_heads) *
                                 mapping_.attention_reduction;
            const auto row = source_neuron / per_row;
            const auto within = source_neuron % per_row;
            const auto head = within / mapping_.attention_reduction;
            const auto reduction = within % mapping_.attention_reduction;
            const auto major = (operand == "lhs" || operand == "q" || operand == "k" || operand == "v")
                                 ? mapping_.attention_rows : mapping_.attention_columns;
            index = static_cast<std::size_t>((head * major + row) *
                                             mapping_.attention_reduction + reduction);
        } else if (operand_layout == "row_head_column") {
            const auto per_row = static_cast<std::uint64_t>(mapping_.attention_heads) *
                                 mapping_.attention_columns;
            const auto reduction = source_neuron / per_row;
            const auto within = source_neuron % per_row;
            const auto head = within / mapping_.attention_columns;
            const auto column = within % mapping_.attention_columns;
            index = static_cast<std::size_t>((head * mapping_.attention_columns + column) *
                                             mapping_.attention_reduction + reduction);
        }
        const auto expected_size = timestep_attention ? static_cast<std::size_t>(mapping_.attention_heads) * mapping_.attention_rows * mapping_.attention_reduction
            : operand == "lhs"
            ? static_cast<std::size_t>(mapping_.attention_heads) * mapping_.attention_rows *
                  mapping_.attention_reduction
            : static_cast<std::size_t>(mapping_.attention_heads) * mapping_.attention_columns *
                  mapping_.attention_reduction;
        if (index >= expected_size) throw std::runtime_error(mapping_.id + ": attention source 越界");
        const auto found = target.find(index);
        const auto old = found == target.end() ? 0 : static_cast<int>(found->second);
        const auto sum = old + static_cast<int>(value);
        if (sum < -127 || sum > 127) throw std::runtime_error(mapping_.id + ": attention transient 溢出");
        if (sum == 0) target.erase(index);
        else target[index] = static_cast<std::int8_t>(sum);
        attention_pending_buffers_[buffer] = 1;
        const auto compute = compute_pipeline_.reserve(
            hw_arrival_time, hardware_.core.axon_in_hw_latency);
        return CoreReceiveResult{compute.hw_finish_time,
                                 compute.hw_finish_time - hw_arrival_time,
                                 hardware_.core.axon_in_hw_latency, 0,
                                 compute.hw_wait_latency, 0};
    }

    // SynapseEngine 直接遍历紧凑模板，Data 只累加到下一 timestep 使用的 buffer。
    // 根据 Spatial Pattern / Linear weight, 找到所有受影响 destination neuron
    const auto physical_end = physical_neuron_begin_ + physical_neuron_count_;
    const auto updates = active_weights.connection_type == ConnectionType::Crossbar
        ? SynapseEngine::apply_crossbar(
            weights_, value, address_.global_core -
                (mapping_.pe * hardware_.core.cores_per_pe + mapping_.core),
            destination_axon, physical_neuron_begin_,
            hardware_.core.crossbar_weight_mode == "nonzero_binary",
            [&](std::uint64_t destination, float delta) {
                const auto index = static_cast<std::size_t>(destination - physical_neuron_begin_);
                delayed_buffers_[buffer][index] += delta;
                delayed_pending_[buffer][index] = 1;
            })
        : SynapseEngine::apply_to_physical_range(
        active_weights, source_neuron, value, mapping_, physical_neuron_begin_, physical_end,
        // 同一 neuron 的多次输入先求和，pending 位区分“没有输入”和“输入和为 0”。
        [&](std::uint64_t destination, float delta) {
            const auto physical = mapping_.physical_neuron_index(destination);
            const auto index = static_cast<std::size_t>(physical - physical_neuron_begin_);
            delayed_buffers_[buffer][index] += delta;
            delayed_pending_[buffer][index] = 1;
        }, destination_axon);

    if (active_weights.connection_type == ConnectionType::Crossbar &&
        hardware_.core.crossbar_packet_activates_all_neurons) {
        // Axon broadcast 即使 crossbar bit 为 0 也会启动该 neuron 的 timestep heartbeat。
        for (std::uint64_t local = 0; local < physical_neuron_count_; ++local) {
            const auto physical = physical_neuron_begin_ + local;
            if (weights_.active_neuron.empty() || weights_.active_neuron[physical] != 0 ||
                hardware_.core.process_inactive_neurons_on_crossbar_event) {
                delayed_pending_[buffer][static_cast<std::size_t>(local)] = 1;
            }
        }
    }

    // Spatial Pattern 与 source-major Dense 保持各自的可配置 synapse service cost。
    const auto hardware_type = connection_weights == nullptr && active_weights.op == LayerOp::Linear
                                   ? ConnectionType::Dense
                                   : active_weights.hardware_type;
    const auto synapse_hw_latency = hardware_type == ConnectionType::Crossbar
                                        ? hardware_.core.crossbar_synapse_hw_latency
                                    : hardware_type == ConnectionType::Identity
                                        ? hardware_.core.identity_synapse_hw_latency
                                    : hardware_type == ConnectionType::Dense ||
                                      hardware_type == ConnectionType::GroupedDense
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

CoreReceiveResult Core::receive_local_state(std::uint64_t source_neuron, float value,
                                       const LayerWeights& connection_weights) {
    if (connection_weights.connection_type != ConnectionType::LocalStateBuffer)
        throw std::logic_error(mapping_.id + ": receive_local_state 的 connection 类型错误");
    if (source_neuron >= connection_weights.source_neurons)
        throw std::runtime_error(mapping_.id + ": local state source neuron 越界");
    const auto physical = mapping_.physical_neuron_index(source_neuron);
    if (physical < physical_neuron_begin_ ||
        physical >= physical_neuron_begin_ + physical_neuron_count_) return {};
    const auto wi = connection_weights.identity_weight.size() == 1 ? 0 :
                    connection_weights.identity_weight.size() == mapping_.neurons
                        ? static_cast<std::size_t>(source_neuron)
                        : static_cast<std::size_t>(source_neuron % mapping_.output_channels);
    const auto local = static_cast<std::size_t>(physical - physical_neuron_begin_);
    state_started_ = true;
    delayed_buffers_[next_buffer_][local] += value * connection_weights.identity_weight.at(wi);
    delayed_pending_[next_buffer_][local] = 1;
    return {};
}

CoreNeuronProcessResult Core::process_timestep(std::uint32_t timestep,
                                               SimTime hw_arrival_time) {
    if (timestep != processed_timestep_ + 1) throw std::logic_error(mapping_.id + ": neuron timestep 不连续");
    auto& timestep_buffer = delayed_buffers_[next_buffer_];
    auto& timestep_pending = delayed_pending_[next_buffer_];

    CoreNeuronProcessResult result;
    if (mapping_.neuron_model == "multi_valued_state") {
        // Cx State 风格的多值 local state buffer 仅在本 logical timestep 存活；
        // 它承担 residual 加法，不产生 spike，也不通过 NoC。
        if (!state_started_) {
            // 上游尚未产生第一个 logical frame 时，不得凭 Conv/BN bias 虚构 activation。
            next_buffer_ = (next_buffer_ + 1) % delayed_buffers_.size();
            processed_timestep_ = timestep;
            result.hw_finish_time = hw_arrival_time;
            return result;
        }
        result.local_state_values.resize(static_cast<std::size_t>(physical_neuron_count_));
        for (std::uint64_t local = 0; local < physical_neuron_count_; ++local) {
            const auto physical = physical_neuron_begin_ + local;
            const auto logical = mapping_.logical_neuron_index(physical);
            const auto bias = weights_.bias.empty() ? 0.0F :
                weights_.bias.size() == mapping_.neurons ? weights_.bias[logical] :
                weights_.bias[logical % mapping_.output_channels];
            result.local_state_values[static_cast<std::size_t>(local)] =
                timestep_buffer[static_cast<std::size_t>(local)] + bias;
            timestep_buffer[static_cast<std::size_t>(local)] = 0.0F;
            timestep_pending[static_cast<std::size_t>(local)] = 0;
        }
        next_buffer_ = (next_buffer_ + 1) % delayed_buffers_.size();
        processed_timestep_ = timestep;
        result.hw_finish_time = hw_arrival_time;
        return result;
    }
    if (attention_ && attention_pending_buffers_[next_buffer_] != 0) {
        std::vector<std::int8_t> lhs(
            static_cast<std::size_t>(mapping_.attention_heads) * mapping_.attention_rows *
                mapping_.attention_reduction, 0);
        std::vector<std::int8_t> rhs(
            static_cast<std::size_t>(mapping_.attention_heads) * mapping_.attention_columns *
                mapping_.attention_reduction, 0);
        for (const auto& [index, value] : attention_lhs_buffers_[next_buffer_]) lhs[index] = value;
        for (const auto& [index, value] : attention_rhs_buffers_[next_buffer_]) rhs[index] = value;
        const auto delta = attention_->update(lhs, rhs);
        for (std::size_t local = 0; local < delta.size(); ++local) {
            timestep_buffer[local] += static_cast<float>(delta[local]) *
                                      mapping_.attention_accumulation_scale;
            if (delta[local] != 0) timestep_pending[local] = 1;
        }
        attention_lhs_buffers_[next_buffer_].clear();
        attention_rhs_buffers_[next_buffer_].clear();
        attention_pending_buffers_[next_buffer_] = 0;
        result.attention_updates = attention_->last_updates();
        const auto latency = mapping_.attention_kind == "qk"
                                 ? hardware_.core.qk_attention_hw_latency
                                 : hardware_.core.qkv_attention_hw_latency;
        if (result.attention_updates > std::numeric_limits<SimTime>::max() /
                                           std::max<SimTime>(latency, 1))
            throw std::runtime_error(mapping_.id + ": attention latency 溢出");
        result.hw_attention_service_latency = result.attention_updates * latency;
    }
    if (timestep_attention_ && attention_pending_buffers_[next_buffer_] != 0) {
        const auto n=static_cast<std::size_t>(mapping_.attention_heads)*mapping_.attention_rows*mapping_.attention_reduction;
        std::vector<std::int8_t> q(n),k(n),v(n);
        for(const auto& [i,x]:attention_lhs_buffers_[next_buffer_])q[i]=x;
        for(const auto& [i,x]:attention_rhs_buffers_[next_buffer_])k[i]=x;
        for(const auto& [i,x]:attention_third_buffers_[next_buffer_])v[i]=x;
        const auto a=timestep_attention_->update(q,k,v);
        for(std::size_t i=0;i<a.output.size();++i){timestep_buffer[i]+=a.output[i];if(a.output[i]!=0)timestep_pending[i]=1;}
        attention_lhs_buffers_[next_buffer_].clear();attention_rhs_buffers_[next_buffer_].clear();attention_third_buffers_[next_buffer_].clear();attention_pending_buffers_[next_buffer_]=0;
        result.attention_updates=a.kv_updates+a.q_updates;
        result.kv_attention_updates=a.kv_updates;
        result.q_attention_updates=a.q_updates;
        const auto latency=hardware_.core.qkv_attention_hw_latency;
        result.hw_attention_service_latency=result.attention_updates*latency;
    }
    if (mapping_.post_accumulation_rounding == "nearest_even") {
        for (auto& value : timestep_buffer) value = std::nearbyint(value);
    }
    result.mapped_neurons = weights_.active_neuron.empty() ? physical_neuron_count_ : 0;
    SimTime hw_service_latency = result.hw_attention_service_latency;
    struct RelativeFiring {
        FiredNeuron fired;
        SimTime service_finish_time = 0;
        std::uint32_t local_neuron = 0;
    };
    std::vector<RelativeFiring> relative_firings;
    for (std::uint64_t local_neuron = 0; local_neuron < physical_neuron_count_; ++local_neuron) {
        // state_start_timestep 由数据流 depth 配置：第一个 NIR frame 到达前不能提前
        // 演化；启动后即使本步无 packet，persistent Cx State 也必须 transition。
        if (mapping_.neuron_model == "st_bif" &&
            timestep < mapping_.state_start_timestep) continue;
        const auto index = static_cast<std::size_t>(local_neuron);
        const auto physical_neuron = physical_neuron_begin_ + local_neuron;
        const bool active = weights_.active_neuron.empty() ||
                            weights_.active_neuron[physical_neuron] != 0;
        if (!active && !hardware_.core.process_inactive_neurons_on_crossbar_event) continue;
        if (!weights_.active_neuron.empty() && active) ++result.mapped_neurons;
        const bool pending = timestep_pending[index] != 0;
        if (hardware_.core.neuron_update_mode == "event_activated_catch_up" && !pending) continue;
        const auto neuron = mapping_.logical_neuron_index(physical_neuron);
        auto add_latency = [&](SimTime latency) {
            if (latency > std::numeric_limits<SimTime>::max() - hw_service_latency) {
                throw std::runtime_error("Core neuron processing latency 溢出");
            }
            hw_service_latency += latency;
        };
        add_latency(hardware_.core.soma_access_hw_latency);

        const float configured_bias = weights_.bias.empty()
                               ? 0.0F
                               : weights_.bias.size() == mapping_.neurons
                                     ? weights_.bias[static_cast<std::size_t>(neuron)]
                                     : weights_.bias[static_cast<std::size_t>(
                                           neuron % mapping_.output_channels)];
        const auto elapsed = hardware_.core.neuron_update_mode == "event_activated_catch_up"
                                 ? timestep - last_state_timestep_[index]
                                 : 1U;
        const auto neuron_result = soma_->process_neuron(
            local_neuron, timestep_buffer[index], pending,
            configured_bias * static_cast<float>(elapsed), mapping_.input_scale);
        last_state_timestep_[index] = timestep;
        if (neuron_result.updated) {
            ++result.updated_neurons;
            add_latency(hardware_.core.soma_update_hw_latency);
        }
        if (neuron_result.fired) {
            add_latency(hardware_.core.soma_fire_hw_latency);
            relative_firings.push_back(RelativeFiring{
                FiredNeuron{neuron, neuron_result.fired->value}, hw_service_latency,
                static_cast<std::uint32_t>(local_neuron)});
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
    result.hw_soma_service_latency = hw_service_latency - result.hw_attention_service_latency;
    result.hw_resource_wait_latency = reservation.hw_wait_latency;
    result.firings.reserve(relative_firings.size());
    for (const auto& firing : relative_firings) {
        result.firings.push_back(
            CoreFiringResult{firing.fired, reservation.hw_start_time + firing.service_finish_time,
                             address_.global_core, firing.local_neuron});
    }
    return result;
}

const std::vector<float>& Core::output_scores() const {
    static const std::vector<float> empty;
    return soma_ ? soma_->voltage() : empty;
}

const std::vector<std::uint32_t>& Core::output_fire_counts() const {
    static const std::vector<std::uint32_t> empty;
    return soma_ ? soma_->fire_count() : empty;
}

}  // namespace soma
