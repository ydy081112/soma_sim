#pragma once

#include <cstdint>
#include <string>

namespace soma {

// 目标硬件模拟时间轴的唯一时间类型，固定以 ps 为单位；host latency 使用秒。
using SimTime = std::uint64_t;

constexpr SimTime kPsPerNs = 1'000;
constexpr SimTime kPsPerUs = 1'000'000;
constexpr SimTime kPsPerMs = 1'000'000'000;
constexpr SimTime kPsPerSecond = 1'000'000'000'000ULL;

SimTime parse_time_ps(const std::string& value);

inline std::uint64_t ceil_cycles(SimTime hw_latency_ps, SimTime hw_cycle_ps) {
    return hw_cycle_ps == 0 ? 0 : (hw_latency_ps + hw_cycle_ps - 1) / hw_cycle_ps;
}

}  // namespace soma
