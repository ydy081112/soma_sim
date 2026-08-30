#include "soma/hw/memory.hpp"

namespace soma {

ResourceReservation MemoryResource::read(SimTime hw_arrival_time, SimTime hw_latency) {
    // MVP 的 1RW SRAM 让读写共享同一端口，因此都复用同一张可用时间状态。
    return port_.reserve(hw_arrival_time, hw_latency);
}

ResourceReservation MemoryResource::write(SimTime hw_arrival_time, SimTime hw_latency) {
    return port_.reserve(hw_arrival_time, hw_latency);
}

}  // namespace soma
