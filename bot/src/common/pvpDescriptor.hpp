#ifndef COMMON_PVP_DESCRIPTOR_HPP_
#define COMMON_PVP_DESCRIPTOR_HPP_

#include "common/itemRequirement.hpp"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <memory>
#include <string>
#include <vector>

namespace rl::ai {
class BaseIntelligence;
} // namespace rl::ai

namespace common {

struct PvpDescriptor {
  using PvpId = uint64_t;
  PvpId pvpId;

  // We use player names rather than global ids because if the character teleports, the global id changes.
  std::string player1Name;
  std::string player2Name;

  // These are the positions where each character must stand at the start of the Pvp.
  sro::Position pvpPositionPlayer1;
  sro::Position pvpPositionPlayer2;

  // These are the items which each character has in their inventory at the start of the Pvp.
  std::vector<common::ItemRequirement> itemRequirements;

  std::shared_ptr<rl::ai::BaseIntelligence> player1Intelligence;
  std::shared_ptr<rl::ai::BaseIntelligence> player2Intelligence;
};

} // namespace common

#endif // COMMON_PVP_DESCRIPTOR_HPP_