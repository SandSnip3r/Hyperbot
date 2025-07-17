#ifndef RL_ITEMS_HPP_
#define RL_ITEMS_HPP_

#include <silkroad_lib/scalar_types.hpp>

#include <array>

namespace rl {

constexpr std::array kItemIdsForObservations = {
  sro::scalar_types::ReferenceObjectId{5},
  // sro::scalar_types::ReferenceObjectId{12}
};

} // namespace rl

#endif // RL_ITEMS_HPP_