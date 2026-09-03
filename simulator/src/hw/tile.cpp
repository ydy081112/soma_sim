#include "soma/hw/tile.hpp"

#include <limits>
#include <stdexcept>

namespace soma {

std::uint32_t TileLayout::total_cores() const {
    const auto total = static_cast<std::uint64_t>(hardware_.core.pe_count) *
                       hardware_.core.cores_per_pe;
    if (total > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("physical Core 数超过 uint32 范围");
    }
    return static_cast<std::uint32_t>(total);
}

PhysicalCoreAddress TileLayout::core_address(std::uint32_t global_core) const {
    if (global_core >= total_cores()) throw std::runtime_error("physical Core id 越界");
    const auto chip = global_core / hardware_.core.cores_per_chip;
    const auto chip_core = global_core % hardware_.core.cores_per_chip;
    const auto tile = global_core / hardware_.core.cores_per_tile;
    if (tile >= hardware_.noc.router_count()) {
        throw std::runtime_error("physical Core 所属 Tile 没有对应 Router");
    }
    if (hardware_.core.cores_per_tile > hardware_.core.cores_per_chip ||
        tile >= hardware_.noc.router_count()) {
        throw std::runtime_error("physical Core 所属 Tile 没有对应 Router");
    }
    return PhysicalCoreAddress{global_core, tile, chip,
                               chip_core % hardware_.core.cores_per_tile, tile};
}

}  // namespace soma
