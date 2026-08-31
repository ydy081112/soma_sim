#include "soma/hw/noc/route.hpp"

#include <cstdlib>
#include <stdexcept>

namespace soma {
// 用来找源router的输出口
Port RouteGeometry::output_port(std::uint32_t source, std::uint32_t destination) const {
    // 这里只接受 Manhattan 相邻的一跳；y 增大方向沿用参考实现命名为 North。
    const auto sx = x(source);
    const auto sy = y(source);
    const auto dx = x(destination);
    const auto dy = y(destination);
    if (sy == dy && sx + 1 == dx) return Port::East;
    if (sy == dy && dx + 1 == sx) return Port::West;
    if (sx == dx && sy + 1 == dy) return Port::North;
    if (sx == dx && dy + 1 == sy) return Port::South;
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
