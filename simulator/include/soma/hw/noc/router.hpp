#pragma once

#include "soma/common/types.hpp"
#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
#include "soma/hw/noc/route.hpp"

#include <array>
#include <cstddef>
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
    const std::vector<SimTime>& hw_output_free_times() const {
        return hw_output_free_time_;
    }

private:
    const HardwareConfig& hardware_;
    RouteGeometry geometry_;
    std::vector<SimTime> hw_output_free_time_;
    std::vector<SimTime> hw_link_free_time_;

    std::size_t resource_index(std::uint32_t router, Port port) const;
};

}  // namespace soma
