#pragma once
#include <string_view>
#include <vector>

namespace sir {

// Turns query text into a raw embedding (not necessarily normalised). The SearchEngine
// validates and normalises whatever comes back.
class QueryEncoder {
public:
    virtual std::vector<float> encode(std::string_view text) = 0;
    virtual ~QueryEncoder() = default;
};

}  // namespace sir
