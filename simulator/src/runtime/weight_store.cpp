#include "soma/runtime/weight_store.hpp"

#include "soma/runtime/npz.hpp"

#include <stdexcept>

namespace soma {

WeightStore WeightStore::load(const std::string& path, const MappingConfig& mapping) {
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
        if (!weights.bias.empty() && weights.bias.size() != layer.output_channels) {
            throw std::runtime_error(layer.id + ": bias shape 不匹配");
        }

        if (layer.op == LayerOp::Linear) {
            const auto& array = archive.at(prefix + "_weight");
            weights.dense_weight = array.as_f32();
            if (weights.dense_weight.size() != layer.source_neurons * layer.neurons) {
                throw std::runtime_error(layer.id + ": dense [Cin,Cout] weight shape 不匹配");
            }
        } else {
            weights.spatial.cin = layer.input_channels;
            weights.spatial.cout = layer.output_channels;
            weights.spatial.channelwise = layer.channelwise;
            weights.spatial.weight = archive.at(prefix + "_weight").as_f32();
            weights.spatial.plan_pattern_id = archive.at(prefix + "_plan_pattern_id").as_i32();
            weights.spatial.plan_dst_base = archive.at(prefix + "_plan_dst_base").as_i32();
            weights.spatial.pattern_ptr = archive.at(prefix + "_pattern_ptr").as_i32();
            weights.spatial.pattern_dst_offset = archive.at(prefix + "_pattern_dst_offset").as_i32();
            weights.spatial.pattern_weight_offset = archive.at(prefix + "_pattern_weight_offset").as_i64();
            weights.spatial.validate(layer.source_neurons, layer.neurons);
        }
        store.layers_.emplace(layer.id, std::move(weights));
    }
    return store;
}

const LayerWeights& WeightStore::at(const std::string& layer_id) const {
    const auto it = layers_.find(layer_id);
    if (it == layers_.end()) throw std::runtime_error("没有 layer weights: " + layer_id);
    return it->second;
}

}  // namespace soma
