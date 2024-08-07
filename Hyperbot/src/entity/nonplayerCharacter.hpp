#ifndef ENTITY_NONPLAYER_CHARACTER_HPP_
#define ENTITY_NONPLAYER_CHARACTER_HPP_

#include "character.hpp"

namespace entity {

class NonplayerCharacter : public Character {
public:
  EntityType entityType() const override { return EntityType::kNonplayerCharacter; }
};

} // namespace entity

#endif // ENTITY_NONPLAYER_CHARACTER_HPP_