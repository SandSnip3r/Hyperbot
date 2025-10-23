#ifndef RL_ITEMS_HPP_
#define RL_ITEMS_HPP_

#include <silkroad_lib/scalar_types.hpp>

#include <array>

namespace rl {

constexpr std::array<sro::scalar_types::ReferenceObjectId, 0> kItemIdsForObservations = {
  // sro::scalar_types::ReferenceObjectId{5}, // HP Recovery Potion (Small)
  // sro::scalar_types::ReferenceObjectId{12} // MP Recovery Potion (Small)
};

} // namespace rl

#endif // RL_ITEMS_HPP_