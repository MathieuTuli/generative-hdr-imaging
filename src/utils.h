#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>

namespace utils {

struct Error {
    bool raise{false};
    std::string message;
};
} // namespace utils
#endif
