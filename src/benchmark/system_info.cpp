#include "benchmark/system_info.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <thread>

#include <sched.h>
#include <sys/utsname.h>

#ifndef SIR_CXX_FLAGS
#define SIR_CXX_FLAGS "unknown"
#endif
#ifndef SIR_BUILD_TYPE
#define SIR_BUILD_TYPE "unknown"
#endif

namespace sir {
namespace {

std::string trim(std::string s) {
    const auto b = s.find_first_not_of(" \t\n\"");
    const auto e = s.find_last_not_of(" \t\n\"");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

std::string readFirstLine(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return trim(line);
}

std::string runCommand(const std::string& cmd) {
    std::unique_ptr<FILE, int (*)(FILE*)> pipe(::popen(cmd.c_str(), "r"), ::pclose);
    if (!pipe) return "unknown";
    std::string out;
    char buf[256];
    while (std::fgets(buf, sizeof buf, pipe.get())) out += buf;
    return trim(out);
}

std::size_t statusField(const std::string& key) {  // kB value from /proc/self/status
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line))
        if (line.rfind(key + ":", 0) == 0) {
            std::istringstream is(line.substr(key.size() + 1));
            std::size_t kb = 0;
            is >> kb;
            return kb * 1024;
        }
    return 0;
}

}  // namespace

std::size_t peakRssBytes() { return statusField("VmHWM"); }
std::size_t currentRssBytes() { return statusField("VmRSS"); }

SystemInfo collectSystemInfo(const std::filesystem::path& repoRoot) {
    SystemInfo s;
    std::ifstream cpu("/proc/cpuinfo");
    std::string line;
    std::set<std::string> cores;
    std::string physId;
    while (std::getline(cpu, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(line.substr(0, colon)), val = trim(line.substr(colon + 1));
        if (key == "model name" && s.cpuModel.empty()) s.cpuModel = val;
        if (key == "physical id") physId = val;
        if (key == "core id") cores.insert(physId + ":" + val);
    }
    s.logicalCpus = std::thread::hardware_concurrency();
    s.physicalCores = static_cast<unsigned>(cores.size());
    std::ifstream mem("/proc/meminfo");
    while (std::getline(mem, line))
        if (line.rfind("MemTotal:", 0) == 0) {
            std::istringstream is(line.substr(9));
            std::size_t kb = 0;
            is >> kb;
            s.ramBytes = kb * 1024;
        }
    std::ifstream osr("/etc/os-release");
    while (std::getline(osr, line))
        if (line.rfind("PRETTY_NAME=", 0) == 0) s.os = trim(line.substr(12));
    utsname u{};
    if (::uname(&u) == 0) s.kernel = std::string(u.sysname) + " " + u.release + " " + u.machine;
#if defined(__clang__)
    s.compiler = std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    s.compiler = std::string("GCC ") + __VERSION__;
#endif
    s.buildType = SIR_BUILD_TYPE;
    s.cxxFlags = trim(SIR_CXX_FLAGS);
    const std::string git = "git -C '" + repoRoot.string() + "' ";
    s.gitCommit = runCommand(git + "rev-parse HEAD 2>/dev/null");
    if (!runCommand(git + "status --porcelain --untracked-files=no 2>/dev/null").empty())
        s.gitCommit += "-dirty";
    s.cpuGovernor = readFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof set, &set) == 0) {
        std::string aff;
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &set)) aff += (aff.empty() ? "" : ",") + std::to_string(c);
        s.cpuAffinity = aff;
    }
    s.loadAverage = readFirstLine("/proc/loadavg");
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char ts[32];
    std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    s.timestampUtc = ts;
    return s;
}

}  // namespace sir
