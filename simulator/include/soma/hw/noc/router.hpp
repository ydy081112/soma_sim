#pragma once

#include "soma/common/types.hpp"
#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
#include "soma/hw/noc/route.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <unordered_map>
#include <vector>

namespace soma {

struct NocTiming {
    SimTime hw_arrival_time = 0;
    SimTime hw_traversal_latency = 0;
    SimTime hw_congestion_latency = 0;
    SimTime hw_link_busy_latency = 0;
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
        std::vector<std::deque<FifoEntry>> fifo_queues;
        std::vector<std::uint32_t> next_fifo_by_core;
        SimTime total_processing_service = 0;
    };
    std::unordered_map<std::uint32_t, DestinationFlow> destination_flows_;
    std::uint32_t pending_destination_router_ = 0;
    std::size_t pending_fifo_index_ = 0;
    bool has_pending_fifo_packet_ = false;

    std::size_t resource_index(std::uint32_t router, Port port) const;
    void traverse_hop(NocTiming& result, std::uint32_t source, std::uint32_t destination);
    void traverse_local(NocTiming& result, std::uint32_t destination);
    void traverse_local(NocTiming& result, std::uint32_t destination,
                        std::uint32_t destination_core_offset);
};

}  // namespace soma
