#include "soma/hw/noc/router.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace soma {

RouterResourceTable::RouterResourceTable(const HardwareConfig& hardware)
    // 每个 router 仅保留五个 output/link 可用时刻，不创建大量 Router 对象。
    : hardware_(hardware), geometry_(hardware.noc),
      // index 0~4   → Router 0 的 5 个 port
      // index 5~9   → Router 1 的 5 个 port ... ...
      hw_output_free_time_(hardware.noc.router_count() * 5, 0),
      hw_link_free_time_(hardware.noc.router_count() * 5, 0),
      local_ejection_free_time_(hardware.noc.router_count() *
                                hardware.core.cores_per_pe, 0) {
    destination_flows_.reserve(hardware.noc.router_count());
}

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

NocTiming RouterResourceTable::traverse(
    SimTime hw_send_time, std::uint32_t source_router,
    std::uint32_t destination_router, std::uint32_t destination_core_offset) {
    if (!hardware_.core.input_fifo) {
        return traverse(hw_send_time, source_router, destination_router);
    }
    if (source_router >= hardware_.noc.router_count() ||
        destination_router >= hardware_.noc.router_count() ||
        destination_core_offset >= hardware_.core.cores_per_pe) {
        throw std::runtime_error("input FIFO packet endpoint 越界");
    }

    const auto source_x = geometry_.x(source_router);
    const auto source_y = geometry_.y(source_router);
    const auto destination_x = geometry_.x(destination_router);
    const auto destination_y = geometry_.y(destination_router);
    const auto x_hops = source_x > destination_x ? source_x - destination_x
                                                  : destination_x - source_x;
    const auto y_hops = source_y > destination_y ? source_y - destination_y
                                                  : destination_y - source_y;
    const auto hops = static_cast<std::uint64_t>(x_hops) + y_hops;

    auto& flow = destination_flows_[destination_router];
    if (flow.fifo_queues.empty()) {
        const auto queue_count = static_cast<std::size_t>(hardware_.core.cores_per_pe) *
                                 hardware_.core.fifo_num_per_core;
        flow.fifo_queues.resize(queue_count);
        flow.next_fifo_by_core.resize(hardware_.core.cores_per_pe, 0);
    }
    auto retire = [&](SimTime time) {
        for (auto& queue : flow.fifo_queues) {
            while (!queue.empty() && queue.front().hw_release_time <= time) {
                if (flow.total_processing_service < queue.front().hw_processing_service) {
                    throw std::logic_error("input FIFO service 统计下溢");
                }
                flow.total_processing_service -= queue.front().hw_processing_service;
                queue.pop_front();
            }
        }
    };
    retire(hw_send_time);

    const auto capacity = static_cast<std::size_t>(hops + 1) *
                          hardware_.core.fifo_num_per_core *
                          hardware_.core.fifo_depth_per_core;
    auto occupancy = [&]() {
        std::size_t total = 0;
        for (const auto& queue : flow.fifo_queues) total += queue.size();
        return total;
    };
    if (occupancy() > capacity) {
        SimTime earliest_release = std::numeric_limits<SimTime>::max();
        for (const auto& queue : flow.fifo_queues) {
            if (!queue.empty()) {
                earliest_release = std::min(earliest_release,
                                            queue.front().hw_release_time);
            }
        }
        hw_send_time = std::max(hw_send_time, earliest_release);
        retire(hw_send_time);
    }

    SimTime minimum_network_latency = static_cast<SimTime>(hops + 1) *
                                      hardware_.noc.router_hw_latency();
    minimum_network_latency += static_cast<SimTime>(x_hops) *
        (source_x < destination_x ? hardware_.noc.east_link_hw_latency
                                  : hardware_.noc.west_link_hw_latency);
    minimum_network_latency += static_cast<SimTime>(y_hops) *
        (source_y < destination_y ? hardware_.noc.north_link_hw_latency
                                  : hardware_.noc.south_link_hw_latency);
    const auto flow_service = static_cast<double>(flow.total_processing_service) /
                              static_cast<double>(hops + 1);
    const auto extra_network_latency = flow_service > minimum_network_latency
        ? static_cast<SimTime>(std::ceil(flow_service - minimum_network_latency))
        : 0;

    NocTiming result;
    result.hw_arrival_time = hw_send_time + extra_network_latency;
    result.hw_congestion_latency = extra_network_latency;
    auto x = source_x;
    auto y = source_y;
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
    traverse_local(result, destination_router, destination_core_offset);

    auto& next_fifo = flow.next_fifo_by_core[destination_core_offset];
    pending_fifo_index_ = static_cast<std::size_t>(destination_core_offset) *
                          hardware_.core.fifo_num_per_core + next_fifo;
    next_fifo = (next_fifo + 1) % hardware_.core.fifo_num_per_core;
    pending_destination_router_ = destination_router;
    has_pending_fifo_packet_ = true;
    return result;
}

void RouterResourceTable::record_destination_processing(
    SimTime hw_processing_start_time, SimTime hw_processing_service) {
    if (!hardware_.core.input_fifo) return;
    if (!has_pending_fifo_packet_) {
        throw std::logic_error("缺少待记录的 input FIFO packet");
    }
    auto& flow = destination_flows_.at(pending_destination_router_);
    auto& queue = flow.fifo_queues.at(pending_fifo_index_);
    if (!queue.empty() && hw_processing_start_time < queue.back().hw_release_time) {
        throw std::logic_error("同一 input FIFO 的释放时间必须保持有序");
    }
    if (std::numeric_limits<SimTime>::max() - flow.total_processing_service <
        hw_processing_service) {
        throw std::overflow_error("input FIFO service 统计溢出");
    }
    queue.push_back(FifoEntry{hw_processing_start_time, hw_processing_service});
    flow.total_processing_service += hw_processing_service;
    has_pending_fifo_packet_ = false;
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
    const auto link_busy_hw_latency = hardware_.core.input_fifo
        ? std::max(hardware_.noc.link_busy_hw_latency, hardware_.hw_cycle_time_ps)
        : (hardware_.noc.asynchronous()
               ? std::max(hardware_.noc.link_busy_hw_latency, link_hw_latency)
               : std::max(hardware_.noc.link_busy_hw_latency,
                          hardware_.hw_cycle_time_ps));
    const auto index = resource_index(source, port);
    const auto hw_start_time = std::max({result.hw_arrival_time,
                                         hw_output_free_time_[index],
                                         hw_link_free_time_[index]});
    result.hw_congestion_latency += hw_start_time - result.hw_arrival_time;
    const auto hw_finish_time = hw_start_time + hardware_.noc.router_hw_latency() +
                                link_hw_latency;
    hw_output_free_time_[index] = hardware_.core.input_fifo
        ? hw_start_time + link_busy_hw_latency
        : hw_finish_time;
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

void RouterResourceTable::traverse_local(
    NocTiming& result, std::uint32_t destination,
    std::uint32_t destination_core_offset) {
    // Local ejection 按 physical Core 独立，避免同一 Tile 的 Core 互相伪串行。
    const auto endpoint = static_cast<std::size_t>(destination) *
                          hardware_.core.cores_per_pe + destination_core_offset;
    const auto hw_start_time = std::max(result.hw_arrival_time,
                                        local_ejection_free_time_[endpoint]);
    result.hw_congestion_latency += hw_start_time - result.hw_arrival_time;
    result.hw_arrival_time = hw_start_time + hardware_.noc.router_hw_latency();
    result.hw_traversal_latency += hardware_.noc.router_hw_latency();
    local_ejection_free_time_[endpoint] = result.hw_arrival_time;
}

}  // namespace soma
