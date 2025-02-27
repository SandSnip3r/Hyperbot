#ifndef COMMON_PVP_DESCRIPTOR_HPP_
#define COMMON_PVP_DESCRIPTOR_HPP_

#include "common/itemRequirement.hpp"
#include "rl/ai/baseIntelligence.hpp"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <vector>

namespace common {

struct PvpDescriptor {
  sro::scalar_types::EntityGlobalId player1GlobalId;
  sro::scalar_types::EntityGlobalId player2GlobalId;
  sro::Position pvpPositionPlayer1;
  sro::Position pvpPositionPlayer2;
  std::vector<common::ItemRequirement> itemRequirements;
  rl::ai::BaseIntelligence *player1Intelligence{nullptr};
  rl::ai::BaseIntelligence *player2Intelligence{nullptr};
};

} // namespace common

#endif // COMMON_PVP_DESCRIPTOR_HPP_