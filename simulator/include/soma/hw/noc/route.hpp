#pragma once

#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace soma {

enum class Port : std::uint8_t { Local = 0, North = 1, East = 2, South = 3, West = 4 };

class RouteGeometry {
public:
    explicit RouteGeometry(const HardwareConfig::Noc& noc) : rows_(noc.rows), cols_(noc.cols) {}

    // 与 SANA-FE 相同：连续 tile id 先沿 NoC 高度方向增长，再进入下一列。
    std::uint32_t x(std::uint32_t router) const { return router / rows_; }
    std::uint32_t y(std::uint32_t router) const { return router % rows_; }
    std::uint32_t router(std::uint32_t x, std::uint32_t y) const { return x * rows_ + y; }
    Port output_port(std::uint32_t source, std::uint32_t destination) const;
    void validate(const StaticRoute& route) const;
    static std::string to_string(Port port);

private:
    std::uint32_t rows_;
    std::uint32_t cols_;
};

}  // namespace soma
