#include "soma/runtime/weight_store.hpp"

#include "soma/runtime/npz.hpp"

#include <stdexcept>

namespace soma {

WeightStore WeightStore::load(const std::string& path, const MappingConfig& mapping) {
    // 数组名称由 mapping.weight_prefix 拼出，避免把模型层名硬编码进 runtime。
    const auto archive = NpzArchive::load(path);
    WeightStore store;
    for (const auto& layer : mapping.layers) {
        if (layer.op == LayerOp::Input) continue;
        LayerWeights weights;
        weights.op = layer.op;
        const auto prefix = layer.weight_prefix;
        weights.bias = archive.find(prefix + "_bias") == nullptr
                           ? std::vector<float>{}
                           : archive.at(prefix + "_bias").as_f32();
        if (!weights.bias.empty() && weights.bias.size() != layer.output_channels &&
            weights.bias.size() != layer.neurons) {
            throw std::runtime_error(layer.id + ": bias shape 不匹配");
        }
        if (layer.op == LayerOp::Crossbar) {
            weights.threshold = archive.at(prefix + "_threshold").as_f32();
            weights.active_neuron = archive.at(prefix + "_active").as_u8();
            weights.crossbar_axon_type = archive.at(prefix + "_axon_type").as_u8();
            weights.crossbar_neuron_weight = archive.at(prefix + "_neuron_weight").as_i16();
            weights.crossbar_rows = archive.at(prefix + "_crossbar_rows").as_u64();
            if (weights.threshold.size() != layer.neurons ||
                weights.active_neuron.size() != layer.neurons ||
                weights.crossbar_neuron_weight.size() != layer.neurons * 4) {
                throw std::runtime_error(layer.id + ": crossbar neuron array shape 不匹配");
            }
            const auto cores = layer.aggregate_core_count;
            if (cores == 0 || weights.crossbar_axon_type.size() % cores != 0) {
                throw std::runtime_error(layer.id + ": crossbar axon type shape 不匹配");
            }
            weights.crossbar_axons = static_cast<std::uint32_t>(
                weights.crossbar_axon_type.size() / cores);
            const auto row_count = static_cast<std::size_t>(cores) * weights.crossbar_axons;
            if (row_count == 0 || weights.crossbar_rows.size() % row_count != 0) {
                throw std::runtime_error(layer.id + ": crossbar row shape 不匹配");
            }
            weights.crossbar_words_per_axon = static_cast<std::uint32_t>(
                weights.crossbar_rows.size() / row_count);
        }
        store.layers_.emplace(layer.id, std::move(weights));
    }
    store.connections_.reserve(mapping.connections.size());
    for (const auto& connection : mapping.connections) {
        const auto& source = mapping.layer(connection.from);
        const auto& layer = mapping.layer(connection.to);
        LayerWeights weights;
        weights.op = layer.op;
        weights.connection_type = connection.type;
        weights.hardware_type = connection.hardware_type;
        weights.source_neurons = source.neurons;
        const auto& prefix = connection.weight_prefix;
        if (connection.type == ConnectionType::Crossbar) {
            weights.route_destination_partition = archive.at(
                prefix + "_route_destination_partition").as_i32();
            weights.route_destination_axon = archive.at(
                prefix + "_route_destination_axon").as_u16();
            if (weights.route_destination_partition.size() != source.neurons ||
                weights.route_destination_axon.size() != source.neurons) {
                throw std::runtime_error(connection.from + " -> " + connection.to +
                                         ": crossbar route shape 不匹配");
            }
        } else if (connection.type == ConnectionType::Dense) {
            // Linear 按 [Cin,Cout] 保存，固定 source 后整段 Cout 连续。
            const auto& array = archive.at(prefix + "_weight");
            weights.dense_weight = array.as_f32();
            if (weights.dense_weight.size() != source.neurons * layer.neurons) {
                throw std::runtime_error(layer.id + ": dense [Cin,Cout] weight shape 不匹配");
            }
        } else if (connection.type == ConnectionType::Identity) {
            weights.identity_weight = archive.at(prefix + "_weight").as_f32();
            if (weights.identity_weight.size() != 1 &&
                weights.identity_weight.size() != layer.output_channels &&
                weights.identity_weight.size() != layer.neurons) {
                throw std::runtime_error(layer.id + ": identity weight 必须为 scalar/channel/neuron");
            }
        } else {
            // 下面的 weights.spatial 是从 weights.npz 读出来
            weights.spatial.cin = source.output_channels;
            weights.spatial.cout = layer.output_channels;
            weights.spatial.channelwise = connection.channelwise;
            weights.spatial.weight = archive.at(prefix + "_weight").as_f32();
            weights.spatial.plan_pattern_id = archive.at(prefix + "_plan_pattern_id").as_i32();
            weights.spatial.plan_dst_base = archive.at(prefix + "_plan_dst_base").as_i32();
            weights.spatial.pattern_ptr = archive.at(prefix + "_pattern_ptr").as_i32();
            weights.spatial.pattern_dst_offset = archive.at(prefix + "_pattern_dst_offset").as_i32();
            weights.spatial.pattern_weight_offset = archive.at(prefix + "_pattern_weight_offset").as_i64();
            weights.spatial.validate(source.neurons, layer.neurons);
        }
        store.connections_.push_back(std::move(weights));
    }
    return store;
}

const LayerWeights& WeightStore::connection(std::size_t connection_index) const {
    return connections_.at(connection_index);
}

const LayerWeights& WeightStore::at(const std::string& layer_id) const {
    const auto it = layers_.find(layer_id);
    if (it == layers_.end()) throw std::runtime_error("没有 layer weights: " + layer_id);
    return it->second;
}

}  // namespace soma
