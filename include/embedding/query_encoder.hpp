#pragma once
#include <string_view>

#include "containers/vector.hpp"

namespace sir {

// Turns query text into a raw embedding (not necessarily normalised). The SearchEngine
// validates and normalises whatever comes back.
class QueryEncoder {
public:
    virtual Vector<float> encode(std::string_view text) = 0;
    virtual ~QueryEncoder() = default;
};

}  // namespace sir
