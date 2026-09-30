#pragma once
// Environment capture for benchmark reports, and process memory from /proc/self/status.
#include <cstddef>
#include <filesystem>
#include <string>

namespace sir {

struct SystemInfo {
    std::string cpuModel;
    unsigned logicalCpus = 0;
    unsigned physicalCores = 0;
    std::size_t ramBytes = 0;
    std::string os;
    std::string kernel;
    std::string compiler;
    std::string buildType;
    std::string cxxFlags;
    std::string gitCommit;
    std::string cpuGovernor;
    std::string cpuAffinity;  // CPUs this process may run on
    std::string loadAverage;  // at collection time
    std::string timestampUtc;
};

SystemInfo collectSystemInfo(const std::filesystem::path& repoRoot);
std::size_t peakRssBytes();     // VmHWM
std::size_t currentRssBytes();  // VmRSS

}  // namespace sir
