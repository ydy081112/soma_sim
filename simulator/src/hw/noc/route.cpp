#include "soma/hw/noc/route.hpp"

#include <cstdlib>
#include <stdexcept>

namespace soma {

Port RouteGeometry::output_port(std::uint32_t source, std::uint32_t destination) const {
    // Router id 使用 row-major 编码；这里只接受 Manhattan 相邻的一跳。
    const auto sr = row(source);
    const auto sc = col(source);
    const auto dr = row(destination);
    const auto dc = col(destination);
    if (sr == dr && sc + 1 == dc) return Port::East;
    if (sr == dr && dc + 1 == sc) return Port::West;
    if (sc == dc && sr + 1 == dr) return Port::South;
    if (sc == dc && dr + 1 == sr) return Port::North;
    throw std::runtime_error("static route 包含非相邻 router");
}

void RouteGeometry::validate(const StaticRoute& route) const {
    // 静态 route 在使用前校验，非法跨跳不会被 runtime 默默修正。
    for (const auto router : route.routers) {
        if (router >= rows_ * cols_) throw std::runtime_error("static route router 越界");
    }
    for (std::size_t i = 1; i < route.routers.size(); ++i) {
        (void)output_port(route.routers[i - 1], route.routers[i]);
    }
}

std::string RouteGeometry::to_string(Port port) {
    switch (port) {
        case Port::Local: return "LOCAL";
        case Port::North: return "NORTH";
        case Port::East: return "EAST";
        case Port::South: return "SOUTH";
        case Port::West: return "WEST";
    }
    return "UNKNOWN";
}

}  // namespace soma
