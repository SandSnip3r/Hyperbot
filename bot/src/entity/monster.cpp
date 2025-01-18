#include "monster.hpp"

#include <absl/log/log.h>

#include <stdexcept>

namespace entity {

uint32_t Monster::getMaxHp(const pk2::CharacterData &characterData) const {
  if (!characterData.haveCharacterWithId(refObjId)) {
    throw std::runtime_error("Don't have character data to get max HP");
  }
  const auto &data = characterData.getCharacterById(refObjId);
  uint32_t hp = data.maxHp;
  using RarityRawType = std::underlying_type_t<decltype(rarity)>;
  const auto nonPartyRarity = static_cast<sro::entity::MonsterRarity>(static_cast<RarityRawType>(rarity) & (static_cast<RarityRawType>(sro::entity::MonsterRarity::kPartyFlag)-1));
  switch (nonPartyRarity) {
    case sro::entity::MonsterRarity::kChampion:
      hp *= 2;
      break;
    case sro::entity::MonsterRarity::kGiant:
      hp *= 20;
      break;
    case sro::entity::MonsterRarity::kElite:
      hp *= 30;
      break;
    default:
      LOG(WARNING) << "Asking for max HP of an unknown monster rarity";
      break;
  }
  if (flags::isSet(rarity, sro::entity::MonsterRarity::kPartyFlag)) {
    // Party monsters have a flat x10 across the base
    hp *= 10;
  }
  return hp;
}

} // namespace entity