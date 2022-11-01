#ifndef SRO_ENTITY_H_
#define SRO_ENTITY_H_

#include <ostream>

namespace sro::entity {

enum class MonsterRarity : uint8_t {
  kGeneral = 0,
  kChampion = 1,
  kUnique = 3,
  kGiant = 4,
  kTitan = 5,
  kElite = 6,
  kEliteStrong = 7,
  kUnique2 = 8,
  kPartyFlag = 16,
  kGeneralParty = 16,
  kChampionParty = 17,
  kUniqueParty = 19,
  kGiantParty = 20,
  kTitanParty = 21,
  kEliteParty = 22,
  kUnique2Party = 24
};

enum class ItemRarity : uint8_t {
  kWhite = 0, // common
  kBlue = 1,
  kSox = 2, // rare
  kSet = 3,
  kRareSet = 6,
  kLegend = 7
};

enum class LifeState : uint8_t {
  kEmbryo = 0,
  kAlive = 1,
  kDead = 2,
  kGone = 3
};

std::ostream& operator<<(std::ostream &stream, MonsterRarity rarity);
std::ostream& operator<<(std::ostream &stream, ItemRarity rarity);
std::ostream& operator<<(std::ostream &stream, LifeState rarity);

} // namespace sro::entity

#endif // SRO_ENTITY_H_
