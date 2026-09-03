#pragma once

#include "soma/config/hardware_config.hpp"

#include <cstdint>

namespace soma {

struct PhysicalCoreAddress {
    std::uint32_t global_core = 0;
    std::uint32_t tile = 0;
    std::uint32_t chip = 0;
    std::uint32_t core_within_tile = 0;
    std::uint32_t router = 0;
};

// Tile 只负责物理地址换算；Core/Router 的资源状态仍保存在各自紧凑对象中。
class TileLayout {
public:
    explicit TileLayout(const HardwareConfig& hardware) : hardware_(hardware) {}

    PhysicalCoreAddress core_address(std::uint32_t global_core) const;
    std::uint32_t total_cores() const;

private:
    const HardwareConfig& hardware_;
};

}  // namespace soma
