#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace soma {

struct NpyArray {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> data;
    bool fortran_order = false;

    std::size_t element_count() const;
    std::vector<float> as_f32() const;
    std::vector<std::int32_t> as_i32() const;
    std::vector<std::int64_t> as_i64() const;
};

class NpzArchive {
public:
    static NpzArchive load(const std::string& path);
    const NpyArray& at(const std::string& key) const;
    const NpyArray* find(const std::string& key) const;
    std::vector<std::string> keys() const;

private:
    std::unordered_map<std::string, NpyArray> arrays_;
};

}  // namespace soma

