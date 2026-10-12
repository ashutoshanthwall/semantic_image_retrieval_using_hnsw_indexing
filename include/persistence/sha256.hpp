#pragma once
// SHA-256 (FIPS 180-4), implemented here to verify that an index was built from the exact
// embedding file named in its header, and to checksum index payloads. Tested against the
// NIST example vectors.
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

#include "containers/fixed_array.hpp"

namespace sir {

class Sha256 {
public:
    Sha256();
    void update(std::span<const std::uint8_t> data);
    void update(const void* data, std::size_t n);
    FixedArray<std::uint8_t, 32> finish();

private:
    void block(const std::uint8_t* p);
    FixedArray<std::uint32_t, 8> h_;
    FixedArray<std::uint8_t, 64> buf_{};
    std::size_t bufLen_ = 0;
    std::uint64_t totalBytes_ = 0;
};

std::string toHex(std::span<const std::uint8_t> bytes);
std::string sha256Hex(const void* data, std::size_t n);
std::string sha256File(const std::filesystem::path& path);

}  // namespace sir
