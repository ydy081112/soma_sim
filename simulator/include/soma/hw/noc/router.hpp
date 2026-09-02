#pragma once

#include "soma/common/types.hpp"
#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
#include "soma/hw/noc/route.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <queue>
#include <vector>

namespace soma {

struct NocTiming {
    SimTime hw_arrival_time = 0;
    SimTime hw_traversal_latency = 0;
    SimTime hw_congestion_latency = 0;
    SimTime hw_link_busy_latency = 0;
    SimTime hw_source_blocking_latency = 0;
    std::uint64_t hops = 0;
    std::array<std::uint64_t, 5> port_hops{};
};

// 不实例化 Router 对象；数组下标即 router/output-port 与有向 link 资源。
class RouterResourceTable {
public:
    explicit RouterResourceTable(const HardwareConfig& hardware);
    NocTiming traverse(SimTime hw_arrival_time, const StaticRoute& route);
    NocTiming traverse(SimTime hw_arrival_time, std::uint32_t source_router,
                       std::uint32_t destination_router);
    NocTiming traverse(SimTime hw_arrival_time, std::uint32_t source_router,
                       std::uint32_t destination_router,
                       std::uint32_t destination_core_offset);
    NocTiming traverse(SimTime hw_arrival_time, std::uint32_t source_router,
                       std::uint32_t source_core_offset,
                       std::uint32_t destination_router,
                       std::uint32_t destination_core_offset);
    void record_destination_processing(SimTime hw_processing_start_time,
                                       SimTime hw_processing_service);
    const std::vector<SimTime>& hw_output_free_times() const {
        return hw_output_free_time_;
    }

private:
    const HardwareConfig& hardware_;
    RouteGeometry geometry_;
    std::vector<SimTime> hw_output_free_time_;
    std::vector<SimTime> hw_link_free_time_;
    std::vector<SimTime> local_ejection_free_time_;

    struct FifoEntry {
        SimTime hw_release_time = 0;
        SimTime hw_processing_service = 0;
    };
    struct DestinationFlow {
        // 每个 destination router 只初始化一次 flat ring；热路径只移动 head/size。
        std::vector<FifoEntry> fifo_entries;
        std::vector<std::size_t> fifo_heads;
        std::vector<std::size_t> fifo_sizes;
        std::vector<std::uint32_t> next_fifo_by_core;
        std::size_t fifo_capacity = 0;
        SimTime total_processing_service = 0;
        bool initialized = false;
    };
    std::vector<DestinationFlow> destination_flows_;
    std::uint32_t pending_destination_router_ = 0;
    std::size_t pending_fifo_index_ = 0;
    bool has_pending_fifo_packet_ = false;

    struct InFlight {
        SimTime release = 0;
        SimTime service = 0;
        std::uint32_t source = 0;
        std::uint32_t destination = 0;
        std::uint32_t source_core = 0;
    };
    struct EarlierRelease {
        bool operator()(const InFlight& left, const InFlight& right) const {
            return left.release > right.release;
        }
    };
    std::vector<double> route_density_;
    std::priority_queue<InFlight, std::vector<InFlight>, EarlierRelease> in_flight_;
    InFlight pending_in_flight_;
    double in_flight_service_ = 0.0;
    std::uint64_t in_flight_count_ = 0;
    bool has_pending_in_flight_ = false;

    std::size_t resource_index(std::uint32_t router, Port port) const;
    void traverse_hop(NocTiming& result, std::uint32_t source, std::uint32_t destination);
    void traverse_local(NocTiming& result, std::uint32_t destination);
    void traverse_local(NocTiming& result, std::uint32_t destination,
                        std::uint32_t destination_core_offset);
    void adjust_density(const InFlight& message, double sign);
    double route_density(const InFlight& message) const;
    void retire_in_flight(SimTime time);
};

}  // namespace soma
