#include "soma/hw/hardware_resource.hpp"

#include <algorithm>

namespace soma {

ResourceReservation HardwareResource::reserve(SimTime hw_arrival_time,
                                              SimTime hw_service_latency) {
    // 互斥硬件资源通过 free time 串行化；等待时间自然体现在 start-arrival 中。
    ResourceReservation result;
    result.hw_start_time = std::max(hw_arrival_time, hw_free_time_);
    result.hw_wait_latency = result.hw_start_time - hw_arrival_time;
    result.hw_finish_time = result.hw_start_time + hw_service_latency;
    hw_free_time_ = result.hw_finish_time;
    return result;
}

}  // namespace soma
