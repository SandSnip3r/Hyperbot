#ifndef ENTITY_PLAYER_CHARACTER_HPP_
#define ENTITY_PLAYER_CHARACTER_HPP_

#include "character.hpp"
#include "packet/enums/packetEnums.hpp"

#include <string>

namespace entity {

class PlayerCharacter : public Character {
public:
  std::string name;
  packet::enums::FreePvpMode freePvpMode;
  EntityType entityType() const override { return EntityType::kPlayerCharacter; }
protected:
  std::string toStringImpl(const sro::pk2::GameData *gameData) const override;
};

} // namespace entity

#endif // ENTITY_PLAYER_CHARACTER_HPP_