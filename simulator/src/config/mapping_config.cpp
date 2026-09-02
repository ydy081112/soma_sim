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

ConnectionType parse_connection_type(const std::string& value) {
    if (value == "spatial") return ConnectionType::Spatial;
    if (value == "dense") return ConnectionType::Dense;
    if (value == "identity") return ConnectionType::Identity;
    throw std::runtime_error("不支持的 connection type: " + value);
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
    // YAML 中每个条目描述一层；大层由起始 Core 和连续 physical partitions 紧凑表示。
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
        layer.aggregate_core_count = u32(node, "aggregate_core_count", 0);
        layer.physical_neuron_order = yaml::get_string(node, "physical_neuron_order", "logical");
        layer.weight_prefix = yaml::get_string(node, "weight_prefix", layer.id);
        layer.threshold = static_cast<float>(yaml::get_double(node, "threshold", 1.0));
        layer.leak = static_cast<float>(yaml::get_double(node, "leak", 1.0));
        layer.reset = yaml::get_string(node, "reset", "soft");
        layer.membrane_quantization_step = static_cast<float>(
            yaml::get_double(node, "membrane_quantization_step", 0.0));
        layer.threshold_comparison = yaml::get_string(
            node, "threshold_comparison", "greater_equal");
        layer.virtual_input = yaml::get_bool(node, "virtual_input", false);
        layer.readout = yaml::get_bool(node, "readout", false);
        layer.channelwise = yaml::get_bool(node, "channelwise", false);
        config.layers.push_back(std::move(layer));
    }

    const auto& connections = mapping.at("connections");
    if (!connections.is_sequence()) throw std::runtime_error("mapping.connections 必须是 sequence");
    for (const auto& node : connections.elements()) {
        ConnectionMapping connection;
        connection.index = config.connections.size();
        connection.from = node.at("from").as_string();
        connection.to = node.at("to").as_string();
        connection.type = parse_connection_type(node.at("type").as_string());
        connection.hardware_type = parse_connection_type(
            yaml::get_string(node, "hardware_type", to_string(connection.type)));
        connection.weight_prefix = node.at("weight_prefix").as_string();
        connection.delay = u32(node, "delay", 0);
        connection.channelwise = yaml::get_bool(node, "channelwise", false);
        config.connections.push_back(std::move(connection));
    }

    // Route 保存 layer 级 router 序列供启动校验/debug，CSV route 字段不进入 runtime。
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
    outgoing_connections_.assign(layers.size(), {});
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (!layer_index_.emplace(layers[i].id, i).second) {
            throw std::runtime_error("重复 layer id: " + layers[i].id);
        }
    }
    for (const auto& connection : connections) {
        outgoing_connections_.at(index_of(connection.from)).push_back(connection.index);
    }
    for (std::size_t i = 0; i < routes.size(); ++i) {
        if (!route_index_.emplace(route_key(routes[i].from, routes[i].to), i).second) {
            throw std::runtime_error("重复 static route: " + routes[i].from + " -> " + routes[i].to);
        }
    }
}
void MappingConfig::validate(std::uint32_t router_count) const {
    // 合法性检查
    // 同时检查 layer 引用和 layer 级静态 route 端点。
    if (layers.empty()) throw std::runtime_error("mapping 至少需要一个 layer");
    for (const auto& layer : layers) {
        if (layer.neurons == 0) throw std::runtime_error(layer.id + ": neurons 必须为正");
        if (layer.router >= router_count) throw std::runtime_error(layer.id + ": router 越界");
        if (layer.op != LayerOp::Input && layer.source_neurons == 0) {
            throw std::runtime_error(layer.id + ": source_neurons 必须为正");
        }
        if (layer.physical_neuron_order != "logical" &&
            layer.physical_neuron_order != "channel_major") {
            throw std::runtime_error(layer.id + ": 不支持的 physical_neuron_order");
        }
        if (layer.physical_neuron_order == "channel_major" &&
            (layer.output_channels == 0 || layer.neurons % layer.output_channels != 0)) {
            throw std::runtime_error(layer.id + ": channel_major neuron shape 不合法");
        }
        if (layer.membrane_quantization_step < 0.0F)
            throw std::runtime_error(layer.id + ": membrane_quantization_step 不得为负");
        if (layer.threshold_comparison != "greater" &&
            layer.threshold_comparison != "greater_equal")
            throw std::runtime_error(layer.id + ": threshold_comparison 仅支持 greater/greater_equal");
    }
    for (const auto& connection : connections) {
        const auto& from = layer(connection.from);
        const auto& to = layer(connection.to);
        if (to.op == LayerOp::Input) throw std::runtime_error("connection 不能指向 input layer");
        if (connection.type == ConnectionType::Identity && from.neurons != to.neurons)
            throw std::runtime_error("identity connection 两端 neuron 数必须相同");
        (void)route(connection.from, connection.to);
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

const std::vector<std::size_t>& MappingConfig::outgoing(std::size_t layer) const {
    return outgoing_connections_.at(layer);
}

std::uint64_t LayerMapping::physical_neuron_index(std::uint64_t logical_neuron) const {
    if (logical_neuron >= neurons) throw std::runtime_error(id + ": logical neuron 越界");
    if (physical_neuron_order == "logical") return logical_neuron;
    const auto spatial_neurons = neurons / output_channels;
    const auto spatial = logical_neuron / output_channels;
    const auto channel = logical_neuron % output_channels;
    return channel * spatial_neurons + spatial;
}

std::string to_string(ConnectionType type) {
    switch (type) {
        case ConnectionType::Spatial: return "spatial";
        case ConnectionType::Dense: return "dense";
        case ConnectionType::Identity: return "identity";
    }
    return "unknown";
}

std::uint64_t LayerMapping::logical_neuron_index(std::uint64_t physical_neuron) const {
    if (physical_neuron >= neurons) throw std::runtime_error(id + ": physical neuron 越界");
    if (physical_neuron_order == "logical") return physical_neuron;
    const auto spatial_neurons = neurons / output_channels;
    const auto channel = physical_neuron / spatial_neurons;
    const auto spatial = physical_neuron % spatial_neurons;
    return spatial * output_channels + channel;
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
