#pragma once
#include <cstdint>
#include <vector>
namespace soma {
struct TimestepAttentionResult { std::vector<float> output; std::uint64_t kv_updates=0, q_updates=0; };
// 无 shadow 的 SSA：三个 operand 只在调用的本 logical timestep 存活。
class TimestepSpikeAttention {
public:
 TimestepSpikeAttention(std::uint32_t heads,std::uint32_t rows,std::uint32_t dim,
                        std::uint64_t output_begin,std::uint64_t output_count,float scale);
 TimestepAttentionResult update(const std::vector<std::int8_t>& q,const std::vector<std::int8_t>& k,const std::vector<std::int8_t>& v) const;
private: std::uint32_t h_,r_,d_;std::uint64_t begin_,count_;float scale_;
};
}
