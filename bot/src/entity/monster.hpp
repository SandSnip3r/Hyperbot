#ifndef ENTITY_MONSTER_HPP_
#define ENTITY_MONSTER_HPP_

#include "nonplayerCharacter.hpp"
#include "pk2/characterData.hpp"

#include <silkroad_lib/entity.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <optional>

namespace entity {

class Monster : public NonplayerCharacter {
public:
  uint32_t getMaxHp(const pk2::CharacterData &characterData) const;
  EntityType entityType() const override { return EntityType::kMonster; }
  sro::entity::MonsterRarity rarity;
  std::optional<sro::scalar_types::EntityGlobalId> targetGlobalId;
};

} // namespace entity

#endif // ENTITY_MONSTER_HPP_