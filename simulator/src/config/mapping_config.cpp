#include "soma/config/mapping_config.hpp"

#include "soma/config/simple_yaml.hpp"

#include <stdexcept>

namespace soma {
namespace {

LayerOp parse_op(const std::string& value) {
    if (value == "input") return LayerOp::Input;
    if (value == "conv2d") return LayerOp::Conv2d;
    if (value == "linear") return LayerOp::Linear;
    if (value == "avgpool2d") return LayerOp::AvgPool2d;
    throw std::runtime_error("不支持的 layer op: " + value);
}

std::uint32_t u32(const yaml::Node& node, const std::string& key, std::uint32_t fallback) {
    return static_cast<std::uint32_t>(yaml::get_u64(node, key, fallback));
}

std::string route_key(const std::string& from, const std::string& to) {
    return from + "\n" + to;
}

}  // namespace

MappingConfig MappingConfig::load(const std::string& path) {
    const auto root = yaml::load_file(path);
    const auto& mapping = root.at("mapping");
    MappingConfig config;
    config.model = yaml::get_string(mapping, "model", "unnamed");
    const auto& layers = mapping.at("layers");
    if (!layers.is_sequence()) throw std::runtime_error("mapping.layers 必须是 sequence");
    // 提取yaml
    // YAML 中每个条目对应一个 layer partition；aggregated 是大层的快速估计模式。
    for (const auto& node : layers.elements()) {
        LayerMapping layer;
        layer.index = config.layers.size();
        layer.id = node.at("id").as_string();
        layer.partition = yaml::get_string(node, "partition", "0");
        layer.op = parse_op(node.at("op").as_string());
        layer.pe = u32(node, "pe", 0);
        layer.core = u32(node, "core", 0);
        layer.router = u32(node, "router", layer.pe);
        layer.neurons = yaml::get_u64(node, "neurons", 0);
        layer.source_neurons = yaml::get_u64(node, "source_neurons", 0);
        layer.input_h = u32(node, "input_h", 1);
        layer.input_w = u32(node, "input_w", 1);
        layer.input_channels = u32(node, "input_channels", static_cast<std::uint32_t>(layer.source_neurons));
        layer.output_h = u32(node, "output_h", 1);
        layer.output_w = u32(node, "output_w", 1);
        layer.output_channels = u32(node, "output_channels", static_cast<std::uint32_t>(layer.neurons));
        layer.weight_prefix = yaml::get_string(node, "weight_prefix", layer.id);
        layer.next = yaml::get_string(node, "next", "");
        layer.threshold = static_cast<float>(yaml::get_double(node, "threshold", 1.0));
        layer.leak = static_cast<float>(yaml::get_double(node, "leak", 1.0));
        layer.reset = yaml::get_string(node, "reset", "soft");
        layer.virtual_input = yaml::get_bool(node, "virtual_input", false);
        layer.readout = yaml::get_bool(node, "readout", false);
        layer.channelwise = yaml::get_bool(node, "channelwise", false);
        config.layers.push_back(std::move(layer));
    }

    // Route 保存完整 router 序列，CSV 中的 route/debug 字段不会进入这里。
    const auto& routes = mapping.at("routes");
    if (!routes.is_sequence()) throw std::runtime_error("mapping.routes 必须是 sequence");
    for (const auto& node : routes.elements()) {
        StaticRoute route;
        route.from = node.at("from").as_string();
        route.to = node.at("to").as_string();
        for (const auto& item : node.at("routers").as_flow_sequence()) {
            route.routers.push_back(static_cast<std::uint32_t>(std::stoul(item)));
        }
        config.routes.push_back(std::move(route));
    }
    config.rebuild_indices();
    return config;
}

void MappingConfig::rebuild_indices() {
    // 热路径只做哈希查询，避免每枚 spike 线性搜索 layer 或 route。
    layer_index_.clear();
    route_index_.clear();
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (!layer_index_.emplace(layers[i].id, i).second) {
            throw std::runtime_error("重复 layer id: " + layers[i].id);
        }
    }
    for (std::size_t i = 0; i < routes.size(); ++i) {
        if (!route_index_.emplace(route_key(routes[i].from, routes[i].to), i).second) {
            throw std::runtime_error("重复 static route: " + routes[i].from + " -> " + routes[i].to);
        }
    }
}
void MappingConfig::validate(std::uint32_t router_count) const {
    // 合法性检查
    // 同时检查 layer 引用和静态 route 端点，保证 mapping 是路由唯一真值来源。
    if (layers.empty()) throw std::runtime_error("mapping 至少需要一个 layer");
    for (const auto& layer : layers) {
        if (layer.neurons == 0) throw std::runtime_error(layer.id + ": neurons 必须为正");
        if (layer.router >= router_count) throw std::runtime_error(layer.id + ": router 越界");
        if (layer.op != LayerOp::Input && layer.source_neurons == 0) {
            throw std::runtime_error(layer.id + ": source_neurons 必须为正");
        }
        if (!layer.next.empty()) {
            (void)this->layer(layer.next);
            (void)this->route(layer.id, layer.next);
        }
    }
    for (const auto& route : routes) {
        const auto& from = layer(route.from);
        const auto& to = layer(route.to);
        if (route.routers.empty()) throw std::runtime_error("static route 不得为空");
        if (route.routers.front() != from.router || route.routers.back() != to.router) {
            throw std::runtime_error("static route 端点与 layer router 不一致: " + route.from + " -> " + route.to);
        }
        for (const auto router : route.routers) {
            if (router >= router_count) throw std::runtime_error("static route router 越界");
        }
    }
}

const LayerMapping& MappingConfig::layer(const std::string& id) const {
    return layers.at(index_of(id));
}

std::size_t MappingConfig::index_of(const std::string& id) const {
    const auto it = layer_index_.find(id);
    if (it == layer_index_.end()) throw std::runtime_error("mapping 不存在 layer: " + id);
    return it->second;
}

const StaticRoute& MappingConfig::route(const std::string& from, const std::string& to) const {
    const auto it = route_index_.find(route_key(from, to));
    if (it == route_index_.end()) throw std::runtime_error("mapping 不存在 route: " + from + " -> " + to);
    return routes.at(it->second);
}

std::string to_string(LayerOp op) {
    switch (op) {
        case LayerOp::Input: return "input";
        case LayerOp::Conv2d: return "conv2d";
        case LayerOp::Linear: return "linear";
        case LayerOp::AvgPool2d: return "avgpool2d";
    }
    return "unknown";
}

}  // namespace soma
