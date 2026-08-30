#include "soma/input_encoder.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace soma {
namespace {

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

std::vector<std::string> split_csv(const std::string& line) {
    // 处理引号和双引号转义，避免 route/debug 字段中的逗号破坏列对齐。
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else quoted = !quoted;
        } else if (c == ',' && !quoted) {
            fields.push_back(trim(current));
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (quoted) throw std::runtime_error("CSV 引号未闭合");
    fields.push_back(trim(current));
    return fields;
}

}  // namespace

InputSpikeFile load_input_spikes_csv(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("无法打开 input spike CSV: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("input spike CSV 为空");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto header = split_csv(line);
    std::unordered_map<std::string, std::size_t> column;
    for (std::size_t i = 0; i < header.size(); ++i) column.emplace(header[i], i);
    // src/dst router 等 trace 列是可选调试信息，routing 仍只读取 mapping.yaml。
    for (const auto* required : {
             "generated_time", "spike_id", "timestep", "layer_id", "src_neuron", "value"}) {
        if (column.find(required) == column.end()) throw std::runtime_error("CSV 缺少列: " + std::string(required));
    }
    auto field = [&](const std::vector<std::string>& row, const std::string& name) -> const std::string& {
        const auto index = column.at(name);
        if (index >= row.size()) throw std::runtime_error("CSV 行缺少字段: " + name);
        return row[index];
    };

    InputSpikeFile result;
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (trim(line).empty()) continue;
        try {
            const auto row = split_csv(line);
            InputSpikeRecord record;
            record.generated_time = parse_time_ps(field(row, "generated_time"));
            record.spike_id = std::stoull(field(row, "spike_id"));
            record.layer_id = field(row, "layer_id");
            record.source_neuron = std::stoull(field(row, "src_neuron"));
            record.value = std::stof(field(row, "value"));
            record.timestep = static_cast<std::uint32_t>(std::stoul(field(row, "timestep")));
            if (record.timestep == 0) {
                throw std::runtime_error("timestep 必须从 1 开始");
            }
            result.last_timestep = std::max(result.last_timestep, record.timestep);
            const auto expected = column.find("expected_output");
            if (expected != column.end() && expected->second < row.size() && !row[expected->second].empty()) {
                const int value = std::stoi(row[expected->second]);
                if (result.expected_output && *result.expected_output != value) {
                    throw std::runtime_error("CSV expected_output 不一致");
                }
                result.expected_output = value;
            }
            result.spikes.push_back(std::move(record));
        } catch (const std::exception& error) {
            throw std::runtime_error("CSV 行 " + std::to_string(line_number) + ": " + error.what());
        }
    }
    if (result.spikes.empty()) throw std::runtime_error("input spike CSV 没有 spike");
    // 逻辑 timestep 是同步注入的分组键；同 timestep 内仍保留 CSV 原始顺序。
    std::stable_sort(result.spikes.begin(), result.spikes.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.timestep < rhs.timestep;
    });
    return result;
}

}  // namespace soma
