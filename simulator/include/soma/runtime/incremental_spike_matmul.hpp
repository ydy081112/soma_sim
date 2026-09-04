#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace soma {

// Loihi-like attention Core 的本地 Cx State。它只保存累计 tracer/shadow，
// 不保存历史 spike；一个实例对应一个已经 mapping 的 destination partition。
class IncrementalSpikeMatmul {
public:
    enum class Kind { Qk, Av };

    IncrementalSpikeMatmul(Kind kind, std::uint32_t heads, std::uint32_t rows,
                           std::uint32_t reduction, std::uint32_t columns,
                           std::uint64_t output_begin = 0, std::uint64_t output_count = 0,
                           bool row_head_column = false);

    // lhs/rhs 是同一个 logical timestep 对齐后的稠密视图。runtime 可由 sparse
    // transient packet 填充该视图，调用返回本 timestep 对 destination 的增量。
    std::vector<std::int32_t> update(const std::vector<std::int8_t>& lhs,
                                     const std::vector<std::int8_t>& rhs);
    const std::vector<std::int32_t>& accumulated_output() const { return output_; }
    void reset();
    std::uint64_t last_updates() const { return last_updates_; }

private:
    Kind kind_;
    std::uint32_t heads_, rows_, reduction_, columns_;
    std::vector<std::int32_t> shadow_lhs_, shadow_rhs_, output_;
    std::vector<std::uint64_t> lhs_keys_, rhs_keys_;
    std::vector<std::uint32_t> output_lhs_slot_, output_rhs_slot_;
    std::vector<std::uint32_t> output_heads_, output_rows_, output_columns_;
    std::uint64_t output_begin_ = 0, output_count_ = 0, last_updates_ = 0;
    bool row_head_column_ = false;
};

}  // namespace soma
