#include "soma/runtime/npz.hpp"

#include <zlib.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>

namespace soma {
namespace {

std::uint16_t u16(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + 2 > bytes.size()) throw std::runtime_error("NPZ 截断");
    return static_cast<std::uint16_t>(bytes[offset]) |
           static_cast<std::uint16_t>(bytes[offset + 1] << 8U);
}

std::uint32_t u32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + 4 > bytes.size()) throw std::runtime_error("NPZ 截断");
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
}

std::vector<std::uint8_t> inflate_raw(const std::uint8_t* data, std::size_t compressed,
                                      std::size_t uncompressed) {
    std::vector<std::uint8_t> out(uncompressed);
    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(data));
    stream.avail_in = static_cast<uInt>(compressed);
    stream.next_out = reinterpret_cast<Bytef*>(out.data());
    stream.avail_out = static_cast<uInt>(out.size());
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) throw std::runtime_error("zlib 初始化失败");
    const int status = inflate(&stream, Z_FINISH);
    inflateEnd(&stream);
    if (status != Z_STREAM_END || stream.total_out != uncompressed) {
        throw std::runtime_error("NPZ deflate 解压失败");
    }
    return out;
}

NpyArray parse_npy(const std::vector<std::uint8_t>& bytes) {
    static const std::uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (bytes.size() < 10 || !std::equal(std::begin(magic), std::end(magic), bytes.begin())) {
        throw std::runtime_error("NPZ entry 不是 NPY");
    }
    const std::uint8_t major = bytes[6];
    std::size_t header_offset = 0;
    std::size_t header_size = 0;
    if (major == 1) {
        header_size = u16(bytes, 8);
        header_offset = 10;
    } else if (major == 2 || major == 3) {
        header_size = u32(bytes, 8);
        header_offset = 12;
    } else {
        throw std::runtime_error("不支持的 NPY version");
    }
    if (header_offset + header_size > bytes.size()) throw std::runtime_error("NPY header 截断");
    const std::string header(reinterpret_cast<const char*>(bytes.data() + header_offset), header_size);
    std::smatch match;
    NpyArray array;
    if (!std::regex_search(header, match, std::regex("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]"))) {
        throw std::runtime_error("NPY header 缺少 descr");
    }
    array.dtype = match[1].str();
    if (!std::regex_search(header, match, std::regex("['\"]fortran_order['\"]\\s*:\\s*(True|False)"))) {
        throw std::runtime_error("NPY header 缺少 fortran_order");
    }
    array.fortran_order = match[1].str() == "True";
    if (array.fortran_order) throw std::runtime_error("不支持 Fortran-order NPY");
    if (!std::regex_search(header, match, std::regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)"))) {
        throw std::runtime_error("NPY header 缺少 shape");
    }
    const std::string shape_text = match[1].str();
    std::regex number("([0-9]+)");
    for (auto it = std::sregex_iterator(shape_text.begin(), shape_text.end(), number);
         it != std::sregex_iterator(); ++it) {
        array.shape.push_back(static_cast<std::size_t>(std::stoull((*it)[1].str())));
    }
    const auto data_offset = header_offset + header_size;
    array.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_offset), bytes.end());
    return array;
}

template <typename T>
std::vector<T> copy_exact(const NpyArray& array, const std::string& dtype) {
    if (array.dtype != dtype && array.dtype != dtype.substr(1)) {
        throw std::runtime_error("NPY dtype 不匹配，期望 " + dtype + "，实际 " + array.dtype);
    }
    if (array.data.size() != array.element_count() * sizeof(T)) throw std::runtime_error("NPY data 大小不匹配");
    std::vector<T> result(array.element_count());
    std::memcpy(result.data(), array.data.data(), array.data.size());
    return result;
}

}  // namespace

std::size_t NpyArray::element_count() const {
    std::size_t count = 1;
    for (const auto dim : shape) {
        if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("NPY shape 溢出");
        }
        count *= dim;
    }
    return count;
}

std::vector<float> NpyArray::as_f32() const {
    if (dtype == "<f4" || dtype == "=f4" || dtype == "f4") return copy_exact<float>(*this, "<f4");
    if (dtype == "<f8" || dtype == "=f8" || dtype == "f8") {
        const auto source = copy_exact<double>(*this, "<f8");
        return std::vector<float>(source.begin(), source.end());
    }
    throw std::runtime_error("无法将 dtype 转成 float32: " + dtype);
}

std::vector<std::int32_t> NpyArray::as_i32() const {
    return copy_exact<std::int32_t>(*this, "<i4");
}

std::vector<std::int64_t> NpyArray::as_i64() const {
    if (dtype == "<i8" || dtype == "=i8" || dtype == "i8") return copy_exact<std::int64_t>(*this, "<i8");
    if (dtype == "<i4" || dtype == "=i4" || dtype == "i4") {
        const auto source = copy_exact<std::int32_t>(*this, "<i4");
        return std::vector<std::int64_t>(source.begin(), source.end());
    }
    throw std::runtime_error("无法将 dtype 转成 int64: " + dtype);
}

NpzArchive NpzArchive::load(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("无法打开 NPZ: " + path);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (bytes.size() < 22) throw std::runtime_error("NPZ 文件过短");

    std::size_t eocd = std::string::npos;
    const std::size_t lower = bytes.size() > 65'557 ? bytes.size() - 65'557 : 0;
    for (std::size_t pos = bytes.size() - 22;; --pos) {
        if (u32(bytes, pos) == 0x06054b50U) { eocd = pos; break; }
        if (pos == lower) break;
    }
    if (eocd == std::string::npos) throw std::runtime_error("NPZ 缺少 ZIP central directory");
    const std::uint16_t entries = u16(bytes, eocd + 10);
    std::size_t cursor = u32(bytes, eocd + 16);
    NpzArchive archive;
    for (std::uint16_t entry = 0; entry < entries; ++entry) {
        if (u32(bytes, cursor) != 0x02014b50U) throw std::runtime_error("NPZ central entry 损坏");
        const auto method = u16(bytes, cursor + 10);
        const auto compressed = u32(bytes, cursor + 20);
        const auto uncompressed = u32(bytes, cursor + 24);
        const auto name_len = u16(bytes, cursor + 28);
        const auto extra_len = u16(bytes, cursor + 30);
        const auto comment_len = u16(bytes, cursor + 32);
        const auto local_offset = u32(bytes, cursor + 42);
        const std::string filename(reinterpret_cast<const char*>(bytes.data() + cursor + 46), name_len);
        cursor += 46 + name_len + extra_len + comment_len;

        if (u32(bytes, local_offset) != 0x04034b50U) throw std::runtime_error("NPZ local entry 损坏");
        const auto local_name_len = u16(bytes, local_offset + 26);
        const auto local_extra_len = u16(bytes, local_offset + 28);
        const auto data_offset = static_cast<std::size_t>(local_offset) + 30 + local_name_len + local_extra_len;
        if (data_offset + compressed > bytes.size()) throw std::runtime_error("NPZ entry 截断");
        std::vector<std::uint8_t> raw;
        if (method == 0) {
            raw.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_offset),
                       bytes.begin() + static_cast<std::ptrdiff_t>(data_offset + compressed));
        } else if (method == 8) {
            raw = inflate_raw(bytes.data() + data_offset, compressed, uncompressed);
        } else {
            throw std::runtime_error("NPZ 使用了不支持的 ZIP compression method");
        }
        std::string key = filename;
        if (key.size() > 4 && key.substr(key.size() - 4) == ".npy") key.resize(key.size() - 4);
        archive.arrays_.emplace(std::move(key), parse_npy(raw));
    }
    return archive;
}

const NpyArray& NpzArchive::at(const std::string& key) const {
    const auto* value = find(key);
    if (value == nullptr) throw std::runtime_error("weights.npz 缺少数组: " + key);
    return *value;
}

const NpyArray* NpzArchive::find(const std::string& key) const {
    const auto it = arrays_.find(key);
    return it == arrays_.end() ? nullptr : &it->second;
}

std::vector<std::string> NpzArchive::keys() const {
    std::vector<std::string> result;
    result.reserve(arrays_.size());
    for (const auto& pair : arrays_) result.push_back(pair.first);
    std::sort(result.begin(), result.end());
    return result;
}

}  // namespace soma

