#ifndef COMMON_RANDOM_HPP_
#define COMMON_RANDOM_HPP_

#include <random>

namespace common {

std::mt19937 createRandomEngine();

} // namespace common

#endif // COMMON_RANDOM_HPP_