#include "soma/hw/noc/router.hpp"

#include <algorithm>
#include <stdexcept>

namespace soma {

RouterResourceTable::RouterResourceTable(const HardwareConfig& hardware)
    // 每个 router 仅保留五个 output/link 可用时刻，不创建大量 Router 对象。
    : hardware_(hardware), geometry_(hardware.noc),
      // index 0~4   → Router 0 的 5 个 port
      // index 5~9   → Router 1 的 5 个 port ... ...
      hw_output_free_time_(hardware.noc.router_count() * 5, 0),
      hw_link_free_time_(hardware.noc.router_count() * 5, 0) {}

std::size_t RouterResourceTable::resource_index(std::uint32_t router, Port port) const {
    return static_cast<std::size_t>(router) * 5 + static_cast<std::size_t>(port);
}

NocTiming RouterResourceTable::traverse(SimTime hw_arrival_time, const StaticRoute& route) {
    geometry_.validate(route);
    NocTiming result;
    result.hw_arrival_time = hw_arrival_time;
    for (std::size_t i = 1; i < route.routers.size(); ++i) {
        traverse_hop(result, route.routers[i - 1], route.routers[i]);
    }
    traverse_local(result, route.routers.back());
    return result;
}

NocTiming RouterResourceTable::traverse(SimTime hw_arrival_time,
                                        std::uint32_t source_router,
                                        std::uint32_t destination_router) {
    if (source_router >= hardware_.noc.router_count() ||
        destination_router >= hardware_.noc.router_count()) {
        throw std::runtime_error("packet router id 越界");
    }
    NocTiming result;
    result.hw_arrival_time = hw_arrival_time;
    auto x = geometry_.x(source_router);
    auto y = geometry_.y(source_router);
    const auto destination_x = geometry_.x(destination_router);
    const auto destination_y = geometry_.y(destination_router);
    // 与参考 detailed scheduler 一致，确定性 XY route 先走 x 再走 y。
    while (x != destination_x) {
        const auto next_x = x < destination_x ? x + 1 : x - 1;
        traverse_hop(result, geometry_.router(x, y), geometry_.router(next_x, y));
        x = next_x;
    }
    while (y != destination_y) {
        const auto next_y = y < destination_y ? y + 1 : y - 1;
        traverse_hop(result, geometry_.router(x, y), geometry_.router(x, next_y));
        y = next_y;
    }
    traverse_local(result, destination_router);
    return result;
}

void RouterResourceTable::traverse_hop(NocTiming& result, std::uint32_t source,
                                       std::uint32_t destination) {
    const auto port = geometry_.output_port(source, destination);
    SimTime link_hw_latency = hardware_.noc.link_hw_latency;
    switch (port) {
        case Port::North: link_hw_latency = hardware_.noc.north_link_hw_latency; break;
        case Port::East: link_hw_latency = hardware_.noc.east_link_hw_latency; break;
        case Port::South: link_hw_latency = hardware_.noc.south_link_hw_latency; break;
        case Port::West: link_hw_latency = hardware_.noc.west_link_hw_latency; break;
        case Port::Local: break;
    }
    const auto link_busy_hw_latency = hardware_.noc.asynchronous()
        ? std::max(hardware_.noc.link_busy_hw_latency, link_hw_latency)
        : std::max(hardware_.noc.link_busy_hw_latency, hardware_.hw_cycle_time_ps);
    const auto index = resource_index(source, port);
    const auto hw_start_time = std::max({result.hw_arrival_time,
                                         hw_output_free_time_[index],
                                         hw_link_free_time_[index]});
    result.hw_congestion_latency += hw_start_time - result.hw_arrival_time;
    const auto hw_finish_time = hw_start_time + hardware_.noc.router_hw_latency() +
                                link_hw_latency;
    hw_output_free_time_[index] = hw_finish_time;
    hw_link_free_time_[index] = hw_start_time + link_busy_hw_latency;
    result.hw_traversal_latency += hardware_.noc.router_hw_latency() + link_hw_latency;
    result.hw_link_busy_latency += link_busy_hw_latency;
    result.hw_arrival_time = hw_finish_time;
    ++result.hops;
    ++result.port_hops[static_cast<std::size_t>(port)];
}

void RouterResourceTable::traverse_local(NocTiming& result, std::uint32_t destination) {
    // 最后一个 router 的 LOCAL output 仍需经过 input-to-output pipeline。
    const auto local_index = resource_index(destination, Port::Local);
    const auto hw_start_time = std::max(result.hw_arrival_time, hw_output_free_time_[local_index]);
    result.hw_congestion_latency += hw_start_time - result.hw_arrival_time;
    result.hw_arrival_time = hw_start_time + hardware_.noc.router_hw_latency();
    result.hw_traversal_latency += hardware_.noc.router_hw_latency();
    hw_output_free_time_[local_index] = result.hw_arrival_time;
}

}  // namespace soma
