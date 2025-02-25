#ifndef COMMON_ITEM_REQUIREMENT_HPP_
#define COMMON_ITEM_REQUIREMENT_HPP_

#include <silkroad_lib/scalar_types.hpp>

#include <cstdint>

namespace common {

struct ItemRequirement {
  sro::scalar_types::ReferenceObjectId refId;
  uint16_t count;
};

} // namespace common

#endif // COMMON_ITEM_REQUIREMENT_HPP_