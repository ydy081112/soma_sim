#include "soma/runtime/incremental_spike_matmul.hpp"

#include <algorithm>
#include <stdexcept>

namespace soma {

IncrementalSpikeMatmul::IncrementalSpikeMatmul(Kind kind, std::uint32_t heads,
                                               std::uint32_t rows,
                                               std::uint32_t reduction,
                                               std::uint32_t columns,
                                               std::uint64_t output_begin,
                                               std::uint64_t output_count,
                                               bool row_head_column)
    : kind_(kind), heads_(heads), rows_(rows), reduction_(reduction), columns_(columns),
      output_begin_(output_begin), row_head_column_(row_head_column) {
    if (heads == 0 || rows == 0 || reduction == 0 || columns == 0)
        throw std::runtime_error("incremental_spike_matmul shape 必须为正");
    const auto total = static_cast<std::uint64_t>(heads) * rows * columns;
    output_count_ = output_count == 0 ? total : output_count;
    if (output_begin_ + output_count_ > total)
        throw std::runtime_error("incremental_spike_matmul output partition 越界");
    output_.assign(static_cast<std::size_t>(output_count_), 0);
    std::unordered_map<std::uint64_t, std::uint32_t> lhs_slots, rhs_slots;
    output_lhs_slot_.reserve(output_.size()); output_rhs_slot_.reserve(output_.size());
    output_heads_.reserve(output_.size()); output_rows_.reserve(output_.size());
    output_columns_.reserve(output_.size());
    for (std::uint64_t local = 0; local < output_count_; ++local) {
        const auto global = output_begin_ + local;
        const auto h = row_head_column_
            ? static_cast<std::uint32_t>((global / columns_) % heads_)
            : static_cast<std::uint32_t>(global / (rows_ * columns_));
        const auto r = row_head_column_
            ? static_cast<std::uint32_t>(global / (heads_ * columns_))
            : static_cast<std::uint32_t>((global %
                (static_cast<std::uint64_t>(rows_) * columns_)) / columns_);
        const auto c = static_cast<std::uint32_t>(global % columns_);
        const auto lhs_key = static_cast<std::uint64_t>(h) * rows_ + r;
        const auto rhs_key = static_cast<std::uint64_t>(h) * columns_ + c;
        auto [lit, linserted] = lhs_slots.emplace(lhs_key, lhs_slots.size());
        if (linserted) lhs_keys_.push_back(lhs_key);
        auto [rit, rinserted] = rhs_slots.emplace(rhs_key, rhs_slots.size());
        if (rinserted) rhs_keys_.push_back(rhs_key);
        output_lhs_slot_.push_back(lit->second); output_rhs_slot_.push_back(rit->second);
        output_heads_.push_back(h); output_rows_.push_back(r); output_columns_.push_back(c);
    }
    shadow_lhs_.assign(lhs_keys_.size() * reduction_, 0);
    shadow_rhs_.assign(rhs_keys_.size() * reduction_, 0);
}

std::vector<std::int32_t> IncrementalSpikeMatmul::update(
    const std::vector<std::int8_t>& lhs, const std::vector<std::int8_t>& rhs) {
    const auto lhs_size = static_cast<std::size_t>(heads_) * rows_ * reduction_;
    const auto rhs_size = static_cast<std::size_t>(heads_) * columns_ * reduction_;
    if (lhs.size() != lhs_size || rhs.size() != rhs_size)
        throw std::runtime_error("incremental_spike_matmul operand shape 不匹配");
    std::vector<std::int32_t> delta(output_.size(), 0);
    // (S_L+L)(S_R+R)^T-S_LS_R^T = S_LR^T+LS_R^T+LR^T。
    // 这使每步累计结果严格等于当前 shadow 的外积；AV 同理，运行时不检查层名。
    last_updates_ = 0;
    for (std::uint64_t local_output = 0; local_output < output_count_; ++local_output) {
                const auto h = output_heads_[local_output];
                const auto r = output_rows_[local_output];
                const auto c = output_columns_[local_output];
                std::int32_t value = 0;
                for (std::uint32_t k = 0; k < reduction_; ++k) {
                    const auto li = (static_cast<std::size_t>(h) * rows_ + r) * reduction_ + k;
                    const auto ri = (static_cast<std::size_t>(h) * columns_ + c) * reduction_ + k;
                    const auto lsi = static_cast<std::size_t>(output_lhs_slot_[local_output]) * reduction_ + k;
                    const auto rsi = static_cast<std::size_t>(output_rhs_slot_[local_output]) * reduction_ + k;
                    const auto l = static_cast<std::int32_t>(lhs[li]);
                    const auto rr = static_cast<std::int32_t>(rhs[ri]);
                    value += shadow_lhs_[lsi] * rr + l * shadow_rhs_[rsi] + l * rr;
                }
                delta[static_cast<std::size_t>(local_output)] = value;
                output_[static_cast<std::size_t>(local_output)] += value;
                last_updates_ += static_cast<std::uint64_t>(reduction_) * 3;
    }
    for (std::size_t slot = 0; slot < lhs_keys_.size(); ++slot) {
        const auto key = lhs_keys_[slot];
        for (std::uint32_t k = 0; k < reduction_; ++k)
            shadow_lhs_[slot * reduction_ + k] += lhs[key * reduction_ + k];
    }
    for (std::size_t slot = 0; slot < rhs_keys_.size(); ++slot) {
        const auto key = rhs_keys_[slot];
        for (std::uint32_t k = 0; k < reduction_; ++k)
            shadow_rhs_[slot * reduction_ + k] += rhs[key * reduction_ + k];
    }
    return delta;
}

void IncrementalSpikeMatmul::reset() {
    std::fill(shadow_lhs_.begin(), shadow_lhs_.end(), 0);
    std::fill(shadow_rhs_.begin(), shadow_rhs_.end(), 0);
    std::fill(output_.begin(), output_.end(), 0);
    last_updates_ = 0;
}

}  // namespace soma
