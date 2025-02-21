#ifndef COMMON_ITEM_REQUIREMENT_HPP_
#define COMMON_ITEM_REQUIREMENT_HPP_

#include <silkroad_lib/pk2/ref/item.hpp>

#include <cstdint>

namespace common {

struct ItemRequirement {
  sro::pk2::ref::ItemId refId;
  uint16_t count;
};

} // namespace common

#endif // COMMON_ITEM_REQUIREMENT_HPP_