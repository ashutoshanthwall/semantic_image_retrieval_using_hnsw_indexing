#pragma once
// Exception types. Exceptions are used only for exceptional paths: bad input files,
// invalid queries, corrupted indexes. Hot loops never throw.
#include <stdexcept>
#include <string>

namespace sir {

struct ConfigError : std::runtime_error { using std::runtime_error::runtime_error; };
struct DataError : std::runtime_error { using std::runtime_error::runtime_error; };
struct EmbeddingError : std::runtime_error { using std::runtime_error::runtime_error; };
struct QueryError : std::invalid_argument { using std::invalid_argument::invalid_argument; };
struct IndexError : std::runtime_error { using std::runtime_error::runtime_error; };
struct PersistenceError : std::runtime_error { using std::runtime_error::runtime_error; };

}  // namespace sir
