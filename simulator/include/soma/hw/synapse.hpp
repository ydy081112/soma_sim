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
        if (weights.connection_type == ConnectionType::Identity) {
            if (source_neuron >= destination_neurons) throw std::runtime_error("identity source neuron 越界");
            const auto wi = weights.identity_weight.size() == 1 ? 0 : source_neuron;
            update(source_neuron, value * weights.identity_weight.at(wi));
            return 1;
        }
        if (weights.connection_type == ConnectionType::Dense || weights.op == LayerOp::Linear) {
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
        if (weights.connection_type == ConnectionType::Crossbar) {
            if (source_neuron >= weights.route_destination_partition.size()) {
                throw std::runtime_error("crossbar route source neuron 越界");
            }
            const auto partition = weights.route_destination_partition[source_neuron];
            if (partition >= 0) visit(static_cast<std::uint32_t>(partition));
            return;
        }
        const auto partition_count = static_cast<std::uint32_t>(
            (destination.neurons + max_neurons_per_core - 1) / max_neurons_per_core);
        if (weights.connection_type == ConnectionType::Identity) {
            const auto physical = destination.physical_neuron_index(source_neuron);
            visit(static_cast<std::uint32_t>(physical / max_neurons_per_core));
            return;
        }
        if (weights.connection_type == ConnectionType::Dense || weights.op == LayerOp::Linear) {
            for (std::uint32_t partition = 0; partition < partition_count; ++partition) visit(partition);
            return;
        }

        const auto& spatial = weights.spatial;
        const auto configured_sources = weights.source_neurons == 0
                                            ? destination.source_neurons
                                            : weights.source_neurons;
        if (source_neuron >= configured_sources) {
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
                                                 Fn&& update,
                                                 std::uint32_t destination_axon = 0) {
        static_cast<void>(destination_axon);
        if (physical_begin >= physical_end || physical_end > destination.neurons) {
            throw std::runtime_error(destination.id + ": physical Core neuron range 不合法");
        }
        std::uint64_t updates = 0;
        if (weights.connection_type == ConnectionType::Identity) {
            if (source_neuron >= weights.source_neurons) throw std::runtime_error("identity source neuron 越界");
            const auto physical = destination.physical_neuron_index(source_neuron);
            if (physical_begin <= physical && physical < physical_end) {
                std::size_t wi = 0;
                if (weights.identity_weight.size() == destination.output_channels)
                    wi = static_cast<std::size_t>(source_neuron % destination.output_channels);
                else if (weights.identity_weight.size() == destination.neurons)
                    wi = static_cast<std::size_t>(source_neuron);
                update(source_neuron, value * weights.identity_weight.at(wi));
                return 1;
            }
            return 0;
        }
        if (weights.connection_type == ConnectionType::Dense || weights.op == LayerOp::Linear) {
            const auto configured_sources = weights.source_neurons == 0
                                                ? destination.source_neurons
                                                : weights.source_neurons;
            if (source_neuron >= configured_sources) {
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

    template <typename Fn>
    static void for_each_destination_packet(const LayerWeights& weights,
                                            std::uint64_t source_neuron,
                                            const LayerMapping& destination,
                                            std::uint32_t max_neurons_per_core,
                                            Fn&& visit) {
        if (weights.connection_type == ConnectionType::Crossbar) {
            if (source_neuron >= weights.route_destination_partition.size()) {
                throw std::runtime_error("crossbar route source neuron 越界");
            }
            const auto partition = weights.route_destination_partition[source_neuron];
            if (partition >= 0) {
                visit(static_cast<std::uint32_t>(partition),
                      static_cast<std::uint32_t>(weights.route_destination_axon[source_neuron]));
            }
            return;
        }
        for_each_destination_partition(
            weights, source_neuron, destination, max_neurons_per_core,
            [&](std::uint32_t partition) { visit(partition, 0U); });
    }

    template <typename Fn>
    static std::uint64_t apply_crossbar(const LayerWeights& weights, float value,
                                        std::uint32_t destination_partition,
                                        std::uint32_t destination_axon,
                                        std::uint64_t physical_begin,
                                        bool nonzero_binary, Fn&& update) {
        if (destination_axon >= weights.crossbar_axons ||
            destination_partition * weights.crossbar_axons + destination_axon >=
                weights.crossbar_axon_type.size()) {
            throw std::runtime_error("crossbar destination axon 越界");
        }
        const auto row = static_cast<std::size_t>(destination_partition) *
                             weights.crossbar_axons + destination_axon;
        const auto type = weights.crossbar_axon_type[row];
        const auto word_base = row * weights.crossbar_words_per_axon;
        std::uint64_t updates = 0;
        for (std::uint32_t word_index = 0;
             word_index < weights.crossbar_words_per_axon; ++word_index) {
            auto bits = weights.crossbar_rows[word_base + word_index];
            while (bits != 0) {
                const auto bit = static_cast<std::uint32_t>(__builtin_ctzll(bits));
                const auto local = static_cast<std::uint64_t>(word_index) * 64 + bit;
                const auto physical = physical_begin + local;
                const auto weight_index = static_cast<std::size_t>(physical) * 4 + type;
                const auto configured_weight = weights.crossbar_neuron_weight.at(weight_index);
                // 部分 crossbar 模型只把权重表当作连接使能；语义由 hardware YAML 选择。
                const auto effective_weight = nonzero_binary
                                                  ? (configured_weight != 0 ? 1.0F : 0.0F)
                                                  : static_cast<float>(configured_weight);
                update(physical, value * effective_weight);
                bits &= bits - 1;
                ++updates;
            }
        }
        return updates;
    }
};

}  // namespace soma
