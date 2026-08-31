#pragma once

#include "soma/runtime/weight_store.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace soma {

class SynapseEngine {
public:
    // apply 统一 Linear 和 Spatial 两条热路径，并通过回调写入 Soma SoA。
    template <typename Fn>
    static std::uint64_t apply(const LayerWeights& weights, std::uint64_t source_neuron,
                               float value, std::uint64_t destination_neurons, Fn&& update) {
        if (weights.op == LayerOp::Linear) {
            // source-major [Cin,Cout] 让一个 source spike 顺序读取一整行权重。
            const auto source_neurons = weights.dense_weight.size() / destination_neurons;
            if (source_neuron >= source_neurons) throw std::runtime_error("dense source neuron 越界");
            const auto base = source_neuron * destination_neurons;
            for (std::uint64_t destination = 0; destination < destination_neurons; ++destination) {
                update(destination, value * weights.dense_weight[base + destination]);
            }
            return destination_neurons;
        }
        return weights.spatial.for_each_destination(source_neuron, value, std::forward<Fn>(update));
    }

    // 只生成 source neuron 实际触达的 destination Core id，不保存任何 synapse 边。
    template <typename Fn>
    static void for_each_destination_partition(const LayerWeights& weights,
                                               std::uint64_t source_neuron,
                                               const LayerMapping& destination,
                                               std::uint32_t max_neurons_per_core,
                                               Fn&& visit) {
        const auto partition_count = static_cast<std::uint32_t>(
            (destination.neurons + max_neurons_per_core - 1) / max_neurons_per_core);
        if (weights.op == LayerOp::Linear) {
            for (std::uint32_t partition = 0; partition < partition_count; ++partition) visit(partition);
            return;
        }

        const auto& spatial = weights.spatial;
        if (source_neuron >= destination.source_neurons) {
            throw std::runtime_error(destination.id + ": packet source neuron 越界");
        }
        const auto source_spatial = static_cast<std::size_t>(source_neuron / spatial.cin);
        const auto source_channel = static_cast<std::size_t>(source_neuron % spatial.cin);
        const auto pattern = static_cast<std::size_t>(spatial.plan_pattern_id.at(source_spatial));
        const auto begin = static_cast<std::size_t>(spatial.pattern_ptr.at(pattern));
        const auto end = static_cast<std::size_t>(spatial.pattern_ptr.at(pattern + 1));
        if (begin == end) return;

        // SANA-FE 的 VGG 空间层按 channel-major 放置，且一个 channel 不超过一个 Core。
        // 此时任一有效空间连接都会触达全部输出 channel 对应的 partitions。
        const auto output_spatial = destination.neurons / destination.output_channels;
        if (!spatial.channelwise && destination.physical_neuron_order == "channel_major" &&
            output_spatial <= max_neurons_per_core &&
            max_neurons_per_core % output_spatial == 0) {
            for (std::uint32_t partition = 0; partition < partition_count; ++partition) visit(partition);
            return;
        }

        std::vector<std::uint8_t> touched(partition_count, 0);
        auto mark = [&](std::uint64_t logical_neuron) {
            const auto physical = destination.physical_neuron_index(logical_neuron);
            touched.at(static_cast<std::size_t>(physical / max_neurons_per_core)) = 1;
        };
        for (std::size_t entry = begin; entry < end; ++entry) {
            const auto destination_spatial = static_cast<std::uint64_t>(
                spatial.plan_dst_base[source_spatial] + spatial.pattern_dst_offset[entry]);
            if (spatial.channelwise) {
                mark(destination_spatial * spatial.cout + source_channel);
            } else {
                for (std::uint32_t channel = 0; channel < spatial.cout; ++channel) {
                    mark(destination_spatial * spatial.cout + channel);
                }
            }
        }
        for (std::uint32_t partition = 0; partition < partition_count; ++partition) {
            if (touched[partition] != 0) visit(partition);
        }
    }

    // 一个 packet 只执行其 destination physical Core 范围内的 local updates。
    template <typename Fn>
    static std::uint64_t apply_to_physical_range(const LayerWeights& weights,
                                                 std::uint64_t source_neuron,
                                                 float value,
                                                 const LayerMapping& destination,
                                                 std::uint64_t physical_begin,
                                                 std::uint64_t physical_end,
                                                 Fn&& update) {
        if (physical_begin >= physical_end || physical_end > destination.neurons) {
            throw std::runtime_error(destination.id + ": physical Core neuron range 不合法");
        }
        std::uint64_t updates = 0;
        if (weights.op == LayerOp::Linear) {
            if (source_neuron >= destination.source_neurons) {
                throw std::runtime_error(destination.id + ": dense source neuron 越界");
            }
            const auto base = source_neuron * destination.neurons;
            for (auto physical = physical_begin; physical < physical_end; ++physical) {
                const auto logical = destination.logical_neuron_index(physical);
                update(logical, value * weights.dense_weight[base + logical]);
                ++updates;
            }
            return updates;
        }

        const auto& spatial = weights.spatial;
        const auto source_spatial = static_cast<std::size_t>(source_neuron / spatial.cin);
        const auto source_channel = static_cast<std::size_t>(source_neuron % spatial.cin);
        const auto pattern = static_cast<std::size_t>(spatial.plan_pattern_id.at(source_spatial));
        const auto entry_begin = static_cast<std::size_t>(spatial.pattern_ptr.at(pattern));
        const auto entry_end = static_cast<std::size_t>(spatial.pattern_ptr.at(pattern + 1));
        const auto output_spatial = destination.neurons / spatial.cout;
        for (std::size_t entry = entry_begin; entry < entry_end; ++entry) {
            const auto destination_spatial = static_cast<std::uint64_t>(
                spatial.plan_dst_base[source_spatial] + spatial.pattern_dst_offset[entry]);
            const auto weight_offset = static_cast<std::size_t>(spatial.pattern_weight_offset[entry]);
            if (spatial.channelwise) {
                const auto logical = destination_spatial * spatial.cout + source_channel;
                const auto physical = destination.physical_neuron_index(logical);
                if (physical_begin <= physical && physical < physical_end) {
                    update(logical, value * spatial.weight.at(weight_offset));
                    ++updates;
                }
                continue;
            }

            const auto weight_base = source_channel * (spatial.weight.size() / spatial.cin) +
                                     weight_offset;
            std::uint64_t first_channel = 0;
            std::uint64_t last_channel = spatial.cout;
            if (destination.physical_neuron_order == "channel_major") {
                first_channel = physical_begin <= destination_spatial
                                    ? 0
                                    : (physical_begin - destination_spatial + output_spatial - 1) /
                                          output_spatial;
                last_channel = physical_end <= destination_spatial
                                   ? 0
                                   : (physical_end - 1 - destination_spatial) / output_spatial + 1;
            } else {
                const auto logical_begin = destination_spatial * spatial.cout;
                const auto logical_end = logical_begin + spatial.cout;
                if (logical_end <= physical_begin || logical_begin >= physical_end) continue;
                first_channel = physical_begin <= logical_begin ? 0 : physical_begin - logical_begin;
                last_channel = std::min<std::uint64_t>(spatial.cout, physical_end - logical_begin);
            }
            first_channel = std::min<std::uint64_t>(first_channel, spatial.cout);
            last_channel = std::min<std::uint64_t>(last_channel, spatial.cout);
            for (auto channel = first_channel; channel < last_channel; ++channel) {
                const auto logical = destination_spatial * spatial.cout + channel;
                update(logical, value * spatial.weight[weight_base + channel]);
                ++updates;
            }
        }
        return updates;
    }
};

}  // namespace soma
