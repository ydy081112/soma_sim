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
    destination_flows_.resize(hardware.noc.router_count());
    if (hardware_.noc.congestion_model == "route_density") {
        route_density_.assign(hardware.noc.router_count() *
                              (4 + hardware.core.cores_per_pe), 0.0);
    }
}

void RouterResourceTable::adjust_density(const InFlight& message, double sign) {
    const auto links = 4 + hardware_.core.cores_per_pe;
    auto add = [&](std::uint32_t router, std::uint32_t link) {
        route_density_.at(static_cast<std::size_t>(router) * links + link) += sign;
    };
    auto x = geometry_.x(message.source);
    auto y = geometry_.y(message.source);
    const auto destination_x = geometry_.x(message.destination);
    const auto destination_y = geometry_.y(message.destination);
    std::uint32_t previous = 4 + message.source_core;
    bool first = true;
    while (x != destination_x) {
        const auto direction = x < destination_x ? 1U : 3U;
        add(geometry_.router(x, y), first ? 4 + message.source_core : direction);
        first = false;
        previous = direction;
        x += x < destination_x ? 1 : -1;
    }
    while (y != destination_y) {
        const auto direction = y < destination_y ? 0U : 2U;
        add(geometry_.router(x, y), first ? 4 + message.source_core : previous);
        first = false;
        previous = direction;
        y += y < destination_y ? 1 : -1;
    }
    add(message.destination, first ? 4 + message.source_core : previous);
}

double RouterResourceTable::route_density(const InFlight& message) const {
    const auto links = 4 + hardware_.core.cores_per_pe;
    auto get = [&](std::uint32_t router, std::uint32_t link) {
        return route_density_.at(static_cast<std::size_t>(router) * links + link);
    };
    double total = 0.0;
    auto x = geometry_.x(message.source);
    auto y = geometry_.y(message.source);
    const auto destination_x = geometry_.x(message.destination);
    const auto destination_y = geometry_.y(message.destination);
    std::uint32_t previous = 4 + message.source_core;
    bool first = true;
    while (x != destination_x) {
        const auto direction = x < destination_x ? 1U : 3U;
        total += get(geometry_.router(x, y), first ? 4 + message.source_core : direction);
        first = false;
        previous = direction;
        x += x < destination_x ? 1 : -1;
    }
    while (y != destination_y) {
        const auto direction = y < destination_y ? 0U : 2U;
        total += get(geometry_.router(x, y), first ? 4 + message.source_core : previous);
        first = false;
        previous = direction;
        y += y < destination_y ? 1 : -1;
    }
    total += get(message.destination, first ? 4 + message.source_core : previous);
    return total;
}

void RouterResourceTable::retire_in_flight(SimTime time) {
    while (!in_flight_.empty() && in_flight_.top().release <= time) {
        const auto message = in_flight_.top();
        in_flight_.pop();
        const auto hops = std::abs(static_cast<int>(geometry_.x(message.source)) -
                                   static_cast<int>(geometry_.x(message.destination))) +
                          std::abs(static_cast<int>(geometry_.y(message.source)) -
                                   static_cast<int>(geometry_.y(message.destination)));
        adjust_density(message, -1.0 / static_cast<double>(hops + 2));
        in_flight_service_ -= static_cast<double>(message.service);
        --in_flight_count_;
    }
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
    if (hardware_.noc.congestion_model == "route_density") {
        return traverse(hw_send_time, source_router, 0, destination_router,
                        destination_core_offset);
    }
    if (!hardware_.core.input_fifo ||
        hardware_.noc.congestion_model == "resource_table") {
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

    auto& flow = destination_flows_.at(destination_router);
    if (!flow.initialized) {
        const auto queue_count = static_cast<std::size_t>(hardware_.core.cores_per_pe) *
                                 hardware_.core.fifo_num_per_core;
        // 预留最大 XY 路径容量加一个入队位置，避免运行时扩容改变回压行为。
        const auto max_hops = static_cast<std::size_t>(hardware_.noc.rows - 1) +
                              static_cast<std::size_t>(hardware_.noc.cols - 1);
        flow.fifo_capacity = (max_hops + 1) * hardware_.core.fifo_num_per_core *
                             hardware_.core.fifo_depth_per_core + 1;
        flow.fifo_entries.resize(queue_count * flow.fifo_capacity);
        flow.fifo_heads.assign(queue_count, 0);
        flow.fifo_sizes.assign(queue_count, 0);
        flow.next_fifo_by_core.resize(hardware_.core.cores_per_pe, 0);
        flow.initialized = true;
    }
    auto retire = [&](SimTime time) {
        for (std::size_t index = 0; index < flow.fifo_heads.size(); ++index) {
            auto& head = flow.fifo_heads[index];
            auto& size = flow.fifo_sizes[index];
            while (size != 0) {
                const auto& entry = flow.fifo_entries[index * flow.fifo_capacity + head];
                if (entry.hw_release_time > time) break;
                if (flow.total_processing_service < entry.hw_processing_service) {
                    throw std::logic_error("input FIFO service 统计下溢");
                }
                flow.total_processing_service -= entry.hw_processing_service;
                head = (head + 1) % flow.fifo_capacity;
                --size;
            }
        }
    };
    retire(hw_send_time);

    const auto capacity = static_cast<std::size_t>(hops + 1) *
                          hardware_.core.fifo_num_per_core *
                          hardware_.core.fifo_depth_per_core;
    auto occupancy = [&]() {
        std::size_t total = 0;
        for (const auto size : flow.fifo_sizes) total += size;
        return total;
    };
    if (occupancy() > capacity) {
        SimTime earliest_release = std::numeric_limits<SimTime>::max();
        for (std::size_t index = 0; index < flow.fifo_heads.size(); ++index) {
            if (flow.fifo_sizes[index] != 0) {
                earliest_release = std::min(
                    earliest_release,
                    flow.fifo_entries[index * flow.fifo_capacity + flow.fifo_heads[index]]
                        .hw_release_time);
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

NocTiming RouterResourceTable::traverse(
    SimTime hw_send_time, std::uint32_t source_router,
    std::uint32_t source_core_offset, std::uint32_t destination_router,
    std::uint32_t destination_core_offset) {
    if (hardware_.noc.congestion_model != "route_density") {
        return traverse(hw_send_time, source_router, destination_router,
                        destination_core_offset);
    }
    if (source_router >= hardware_.noc.router_count() ||
        destination_router >= hardware_.noc.router_count() ||
        source_core_offset >= hardware_.core.cores_per_pe ||
        destination_core_offset >= hardware_.core.cores_per_pe) {
        throw std::runtime_error("route-density packet endpoint 越界");
    }

    retire_in_flight(hw_send_time);
    InFlight message{0, 0, source_router, destination_router, source_core_offset};
    const auto source_x = geometry_.x(source_router);
    const auto source_y = geometry_.y(source_router);
    const auto destination_x = geometry_.x(destination_router);
    const auto destination_y = geometry_.y(destination_router);
    const auto x_hops = source_x > destination_x ? source_x - destination_x
                                                  : destination_x - source_x;
    const auto y_hops = source_y > destination_y ? source_y - destination_y
                                                  : destination_y - source_y;
    const auto hops = static_cast<std::uint64_t>(x_hops) + y_hops;
    const auto density = route_density(message);
    const auto mean_service = in_flight_count_ == 0
        ? 0.0 : in_flight_service_ / static_cast<double>(in_flight_count_);
    const auto capacity = static_cast<double>((hops + 1) *
                                               hardware_.core.fifo_depth_per_core);
    SimTime blocking = 0;
    if (density > capacity) {
        blocking = static_cast<SimTime>(std::ceil((density - capacity) * mean_service));
        hw_send_time += blocking;
    }
    const auto congestion = static_cast<SimTime>(std::ceil(
        density * mean_service / static_cast<double>(hops + 1)));
    SimTime minimum = static_cast<SimTime>(hops + 1) * hardware_.noc.router_hw_latency();
    minimum += static_cast<SimTime>(x_hops) *
        (source_x < destination_x ? hardware_.noc.east_link_hw_latency
                                  : hardware_.noc.west_link_hw_latency);
    minimum += static_cast<SimTime>(y_hops) *
        (source_y < destination_y ? hardware_.noc.north_link_hw_latency
                                  : hardware_.noc.south_link_hw_latency);

    NocTiming result;
    result.hw_arrival_time = hw_send_time + std::max(minimum, congestion);
    result.hw_congestion_latency = blocking +
        (congestion > minimum ? congestion - minimum : 0);
    result.hw_source_blocking_latency = blocking;
    result.hw_traversal_latency = minimum;
    result.hops = hops;
    if (x_hops != 0) result.port_hops[source_x < destination_x ? 2 : 4] += x_hops;
    if (y_hops != 0) result.port_hops[source_y < destination_y ? 1 : 3] += y_hops;
    pending_in_flight_ = message;
    pending_in_flight_.release = result.hw_arrival_time;
    has_pending_in_flight_ = true;
    return result;
}

void RouterResourceTable::record_destination_processing(
    SimTime hw_processing_start_time, SimTime hw_processing_service) {
    if (hardware_.noc.congestion_model == "route_density") {
        if (!has_pending_in_flight_) {
            throw std::logic_error("缺少待记录的 route-density packet");
        }
        pending_in_flight_.release = hw_processing_start_time;
        pending_in_flight_.service = hw_processing_service;
        const auto hops = std::abs(
                              static_cast<int>(geometry_.x(pending_in_flight_.source)) -
                              static_cast<int>(geometry_.x(pending_in_flight_.destination))) +
                          std::abs(
                              static_cast<int>(geometry_.y(pending_in_flight_.source)) -
                              static_cast<int>(geometry_.y(pending_in_flight_.destination)));
        adjust_density(pending_in_flight_, 1.0 / static_cast<double>(hops + 2));
        in_flight_service_ += static_cast<double>(hw_processing_service);
        ++in_flight_count_;
        in_flight_.push(pending_in_flight_);
        has_pending_in_flight_ = false;
        return;
    }
    if (!hardware_.core.input_fifo ||
        hardware_.noc.congestion_model == "resource_table") return;
    if (!has_pending_fifo_packet_) {
        throw std::logic_error("缺少待记录的 input FIFO packet");
    }
    auto& flow = destination_flows_.at(pending_destination_router_);
    auto& head = flow.fifo_heads.at(pending_fifo_index_);
    auto& size = flow.fifo_sizes.at(pending_fifo_index_);
    const auto back = (head + size - 1) % flow.fifo_capacity;
    if (size != 0 && hw_processing_start_time <
                         flow.fifo_entries[pending_fifo_index_ * flow.fifo_capacity + back]
                             .hw_release_time) {
        throw std::logic_error("同一 input FIFO 的释放时间必须保持有序");
    }
    if (std::numeric_limits<SimTime>::max() - flow.total_processing_service <
        hw_processing_service) {
        throw std::overflow_error("input FIFO service 统计溢出");
    }
    if (size == flow.fifo_capacity) {
        throw std::overflow_error("input FIFO ring capacity 不足");
    }
    flow.fifo_entries[pending_fifo_index_ * flow.fifo_capacity +
                      (head + size) % flow.fifo_capacity] =
        FifoEntry{hw_processing_start_time, hw_processing_service};
    ++size;
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
