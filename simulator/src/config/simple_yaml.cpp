#include "soma/config/simple_yaml.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace soma::yaml {
namespace {

struct Line {
    // 词法阶段先保留缩进和原始行号，递归解析时无需再次扫描文件。
    int indent = 0;
    std::size_t number = 0;
    std::string text;
};

std::string trim(const std::string& value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    return first >= last ? std::string{} : std::string(first, last);
}

std::string unquote(std::string value) {
    value = trim(value);
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\''))) {
        value = value.substr(1, value.size() - 2);
    }
    return value;
}

std::size_t find_mapping_colon(const std::string& text) {
    // 引号或 flow collection 内的冒号属于标量内容，不能被当成 map 分隔符。
    bool single = false;
    bool dual = false;
    int brackets = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '\'' && !dual) single = !single;
        if (c == '"' && !single) dual = !dual;
        if (!single && !dual) {
            if (c == '[' || c == '{') ++brackets;
            if (c == ']' || c == '}') --brackets;
            if (c == ':' && brackets == 0) return i;
        }
    }
    return std::string::npos;
}

std::vector<Line> read_lines(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("无法打开 YAML: " + path);
    std::vector<Line> lines;
    std::string raw;
    std::size_t number = 0;
    while (std::getline(input, raw)) {
        ++number;
        if (!raw.empty() && raw.back() == '\r') raw.pop_back();
        // 仅删除引号外的注释，避免截断包含 '#' 的名称或路径。
        bool single = false;
        bool dual = false;
        std::size_t comment = std::string::npos;
        for (std::size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] == '\'' && !dual) single = !single;
            if (raw[i] == '"' && !single) dual = !dual;
            if (raw[i] == '#' && !single && !dual) {
                comment = i;
                break;
            }
        }
        if (comment != std::string::npos) raw.resize(comment);
        if (trim(raw).empty()) continue;
        int indent = 0;
        while (indent < static_cast<int>(raw.size()) && raw[static_cast<std::size_t>(indent)] == ' ') {
            ++indent;
        }
        if (indent < static_cast<int>(raw.size()) && raw[static_cast<std::size_t>(indent)] == '\t') {
            throw std::runtime_error("YAML 不允许 tab 缩进，行 " + std::to_string(number));
        }
        lines.push_back(Line{indent, number, trim(raw.substr(static_cast<std::size_t>(indent)))});
    }
    return lines;
}

Node parse_block(const std::vector<Line>& lines, std::size_t& pos, int indent);

void parse_map_entry(Node& result, const std::vector<Line>& lines, std::size_t& pos,
                     int indent, const std::string& text, std::size_t line_number) {
    const auto colon = find_mapping_colon(text);
    if (colon == std::string::npos) {
        throw std::runtime_error("YAML map 缺少冒号，行 " + std::to_string(line_number));
    }
    const auto key = unquote(text.substr(0, colon));
    const auto rest = trim(text.substr(colon + 1));
    if (key.empty()) throw std::runtime_error("YAML key 为空，行 " + std::to_string(line_number));
    if (!rest.empty()) {
        result.put(key, Node(unquote(rest)));
        return;
    }
    if (pos < lines.size() && lines[pos].indent > indent) {
        result.put(key, parse_block(lines, pos, lines[pos].indent));
    } else {
        result.put(key, Node{});
    }
}

Node parse_block(const std::vector<Line>& lines, std::size_t& pos, int indent) {
    // 同一缩进层只能是一种容器；更深缩进由递归调用消费。
    if (pos >= lines.size()) return Node{};
    const bool sequence = lines[pos].text.rfind("-", 0) == 0 &&
                          (lines[pos].text.size() == 1 || lines[pos].text[1] == ' ');
    Node result = sequence ? Node::sequence() : Node::map();

    while (pos < lines.size() && lines[pos].indent == indent) {
        const Line line = lines[pos++];
        const bool item = line.text.rfind("-", 0) == 0 &&
                          (line.text.size() == 1 || line.text[1] == ' ');
        if (item != sequence) {
            throw std::runtime_error("YAML sequence/map 混用，行 " + std::to_string(line.number));
        }
        if (!sequence) {
            parse_map_entry(result, lines, pos, indent, line.text, line.number);
            continue;
        }

        const auto rest = trim(line.text.substr(1));
        if (rest.empty()) {
            result.append(pos < lines.size() && lines[pos].indent > indent
                              ? parse_block(lines, pos, lines[pos].indent)
                              : Node{});
            continue;
        }
        if (find_mapping_colon(rest) == std::string::npos) {
            result.append(Node(unquote(rest)));
            continue;
        }

        Node map_item = Node::map();
        parse_map_entry(map_item, lines, pos, indent, rest, line.number);
        // “- id: x” 后面的缩进字段仍属于同一个 list item，需要合并进该 map。
        if (pos < lines.size() && lines[pos].indent > indent) {
            const int child_indent = lines[pos].indent;
            Node tail = parse_block(lines, pos, child_indent);
            if (!tail.is_map()) {
                throw std::runtime_error("YAML list map 后只能继续 map，行 " +
                                         std::to_string(line.number));
            }
            for (const auto& pair : tail.items()) map_item.put(pair.first, pair.second);
        }
        result.append(std::move(map_item));
    }
    return result;
}

}  // namespace

Node::Node(std::string scalar) : kind_(Kind::Scalar), scalar_(std::move(scalar)) {}
Node Node::map() { Node n; n.kind_ = Kind::Map; return n; }
Node Node::sequence() { Node n; n.kind_ = Kind::Sequence; return n; }

const Node& Node::at(const std::string& key) const {
    const auto* result = find(key);
    if (result == nullptr) throw std::runtime_error("YAML 缺少字段: " + key);
    return *result;
}

const Node* Node::find(const std::string& key) const {
    if (!is_map()) return nullptr;
    const auto it = map_.find(key);
    return it == map_.end() ? nullptr : &it->second;
}

std::string Node::as_string() const {
    if (!is_scalar()) throw std::runtime_error("YAML 节点不是标量");
    return scalar_;
}

std::int64_t Node::as_i64() const {
    const auto text = as_string();
    std::size_t consumed = 0;
    const auto value = std::stoll(text, &consumed, 0);
    if (consumed != text.size()) throw std::runtime_error("无效整数: " + text);
    return value;
}

std::uint64_t Node::as_u64() const {
    const auto value = as_i64();
    if (value < 0) throw std::runtime_error("期望非负整数");
    return static_cast<std::uint64_t>(value);
}

double Node::as_double() const {
    const auto text = as_string();
    std::size_t consumed = 0;
    const auto value = std::stod(text, &consumed);
    if (consumed != text.size()) throw std::runtime_error("无效浮点数: " + text);
    return value;
}

bool Node::as_bool() const {
    auto text = as_string();
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (text == "true" || text == "yes" || text == "on" || text == "1") return true;
    if (text == "false" || text == "no" || text == "off" || text == "0") return false;
    throw std::runtime_error("无效布尔值: " + text);
}

std::vector<std::string> Node::as_flow_sequence() const {
    if (is_sequence()) {
        std::vector<std::string> out;
        out.reserve(sequence_.size());
        for (const auto& item : sequence_) out.push_back(item.as_string());
        return out;
    }
    auto text = as_string();
    if (text.size() < 2 || text.front() != '[' || text.back() != ']') {
        throw std::runtime_error("期望 flow sequence: " + text);
    }
    text = text.substr(1, text.size() - 2);
    std::vector<std::string> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) out.push_back(unquote(item));
    if (out.size() == 1 && out.front().empty()) out.clear();
    return out;
}

void Node::put(std::string key, Node value) {
    if (!is_map()) throw std::logic_error("put 只能用于 YAML map");
    map_[std::move(key)] = std::move(value);
}

void Node::append(Node value) {
    if (!is_sequence()) throw std::logic_error("append 只能用于 YAML sequence");
    sequence_.push_back(std::move(value));
}

Node load_file(const std::string& path) {
    const auto lines = read_lines(path);
    if (lines.empty()) return Node::map();
    std::size_t pos = 0;
    Node root = parse_block(lines, pos, lines.front().indent);
    if (pos != lines.size()) throw std::runtime_error("YAML 存在无法解析的尾部: " + path);
    return root;
}

std::string get_string(const Node& node, const std::string& key, const std::string& fallback) {
    const auto* value = node.find(key);
    return value == nullptr || value->is_null() ? fallback : value->as_string();
}
std::uint64_t get_u64(const Node& node, const std::string& key, std::uint64_t fallback) {
    const auto* value = node.find(key);
    return value == nullptr || value->is_null() ? fallback : value->as_u64();
}
double get_double(const Node& node, const std::string& key, double fallback) {
    const auto* value = node.find(key);
    return value == nullptr || value->is_null() ? fallback : value->as_double();
}
bool get_bool(const Node& node, const std::string& key, bool fallback) {
    const auto* value = node.find(key);
    return value == nullptr || value->is_null() ? fallback : value->as_bool();
}

}  // namespace soma::yaml
