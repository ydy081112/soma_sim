#include "soma/hw/noc/router.hpp"

#include <algorithm>

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
    const auto router_hw_latency = hardware_.noc.router_hw_latency();
    for (std::size_t i = 1; i < route.routers.size(); ++i) {
        // 每次拿相邻的两个 router，组成一跳
        const auto source = route.routers[i - 1];
        const auto destination = route.routers[i];
        const auto port = geometry_.output_port(source, destination);
        SimTime link_hw_latency = hardware_.noc.link_hw_latency;
        switch (port) {
            case Port::North: link_hw_latency = hardware_.noc.north_link_hw_latency; break;
            case Port::East: link_hw_latency = hardware_.noc.east_link_hw_latency; break;
            case Port::South: link_hw_latency = hardware_.noc.south_link_hw_latency; break;
            case Port::West: link_hw_latency = hardware_.noc.west_link_hw_latency; break;
            case Port::Local: break;
        }
        // req+ack 信号存在时采用配置的握手占用；同步 link 则至少占一个周期。
        const auto link_busy_hw_latency = hardware_.noc.asynchronous()
            ? std::max(hardware_.noc.link_busy_hw_latency, link_hw_latency)
            : std::max(hardware_.noc.link_busy_hw_latency, hardware_.hw_cycle_time_ps);
        const auto index = resource_index(source, port);

        // 当前 hop 必须同时等待 spike 到达、output 空闲和有向 link 空闲。
        const auto hw_start_time = std::max({
            result.hw_arrival_time,
            hw_output_free_time_[index], 
            hw_link_free_time_[index]});
        result.hw_congestion_latency += hw_start_time - result.hw_arrival_time;

        const auto hw_finish_time = hw_start_time + router_hw_latency + link_hw_latency;

        // output 与 link 可用时刻分开维护，便于表达流水 link 或异步握手占用。
        hw_output_free_time_[index] = hw_finish_time;
        hw_link_free_time_[index] = hw_start_time + link_busy_hw_latency;

        result.hw_traversal_latency += router_hw_latency + link_hw_latency;
        result.hw_link_busy_latency += link_busy_hw_latency;
        
        result.hw_arrival_time = hw_finish_time;
        ++result.hops;
        ++result.port_hops[static_cast<std::size_t>(port)];
    }

    // 最后一个 router 的 LOCAL output 仍需经过 input-to-output pipeline。
    const auto destination = route.routers.back();
    const auto local_index = resource_index(destination, Port::Local);
    const auto hw_start_time = std::max(result.hw_arrival_time, hw_output_free_time_[local_index]);
    result.hw_congestion_latency += hw_start_time - result.hw_arrival_time;
    result.hw_arrival_time = hw_start_time + router_hw_latency;
    result.hw_traversal_latency += router_hw_latency;
    hw_output_free_time_[local_index] = result.hw_arrival_time;
    return result;
}

}  // namespace soma
