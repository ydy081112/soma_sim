#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace soma::yaml {

// MVP 使用的 YAML 子集：缩进 map/sequence、标量与 [a, b] flow sequence。
// 配置错误会直接报告字段路径，避免静默采用错误硬件参数。
class Node {
public:
    enum class Kind { Null, Scalar, Map, Sequence };

    Node() = default;
    explicit Node(std::string scalar);
    static Node map();
    static Node sequence();

    Kind kind() const { return kind_; }
    bool is_null() const { return kind_ == Kind::Null; }
    bool is_scalar() const { return kind_ == Kind::Scalar; }
    bool is_map() const { return kind_ == Kind::Map; }
    bool is_sequence() const { return kind_ == Kind::Sequence; }

    const Node& at(const std::string& key) const;
    const Node* find(const std::string& key) const;
    const std::map<std::string, Node>& items() const { return map_; }
    const std::vector<Node>& elements() const { return sequence_; }

    std::string as_string() const;
    std::int64_t as_i64() const;
    std::uint64_t as_u64() const;
    double as_double() const;
    bool as_bool() const;
    std::vector<std::string> as_flow_sequence() const;

    void put(std::string key, Node value);
    void append(Node value);

private:
    Kind kind_ = Kind::Null;
    std::string scalar_;
    std::map<std::string, Node> map_;
    std::vector<Node> sequence_;
};

Node load_file(const std::string& path);

std::string get_string(const Node& node, const std::string& key, const std::string& fallback);
std::uint64_t get_u64(const Node& node, const std::string& key, std::uint64_t fallback);
double get_double(const Node& node, const std::string& key, double fallback);
bool get_bool(const Node& node, const std::string& key, bool fallback);

}  // namespace soma::yaml

