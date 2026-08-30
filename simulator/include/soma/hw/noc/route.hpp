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

    std::uint32_t row(std::uint32_t router) const { return router / cols_; }
    std::uint32_t col(std::uint32_t router) const { return router % cols_; }
    Port output_port(std::uint32_t source, std::uint32_t destination) const;
    void validate(const StaticRoute& route) const;
    static std::string to_string(Port port);

private:
    std::uint32_t rows_;
    std::uint32_t cols_;
};

}  // namespace soma

