#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>

#define ASSERT(condition, fmt, ...) \
    do { \
        if (!(condition)) { \
            char buffer[1024]; \
            std::snprintf(buffer, sizeof(buffer), fmt, ##__VA_ARGS__); \
            std::cerr << "Assertion failed: " << #condition << ", " << buffer \
                      << ", file " << __FILE__ << ", line " << __LINE__ << std::endl; \
            throw std::runtime_error(""); \
        } \
    } while (false)
#define CLIP(x, min, max) ((x) < (min)) ? (min) : ((x) > (max)) ? (max) : (x)
#define CLAMP(x) CLIP(x, 0.0, 1.0)
namespace utils {

struct Error {
    bool raise{false};
    std::string message;
};
} // namespace utils
#endif
