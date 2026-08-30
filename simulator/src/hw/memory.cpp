#include "soma/hw/memory.hpp"

namespace soma {

ResourceReservation MemoryResource::read(SimTime hw_arrival_time, SimTime hw_latency) {
    return port_.reserve(hw_arrival_time, hw_latency);
}

ResourceReservation MemoryResource::write(SimTime hw_arrival_time, SimTime hw_latency) {
    return port_.reserve(hw_arrival_time, hw_latency);
}

}  // namespace soma
