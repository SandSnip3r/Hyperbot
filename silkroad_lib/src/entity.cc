#include "entity.h"

namespace sro::entity {

std::ostream& operator<<(std::ostream &stream, MonsterRarity rarity) {
  switch (rarity) {
    case MonsterRarity::kGeneral:
      stream << "General";
      break;
    case MonsterRarity::kChampion:
      stream << "Champion";
      break;
    case MonsterRarity::kUnique:
      stream << "Unique";
      break;
    case MonsterRarity::kGiant:
      stream << "Giant";
      break;
    case MonsterRarity::kTitan:
      stream << "Titan";
      break;
    case MonsterRarity::kElite:
      stream << "Elite";
      break;
    case MonsterRarity::kEliteStrong:
      stream << "EliteStrong";
      break;
    case MonsterRarity::kUnique2:
      stream << "Unique2";
      break;
    case MonsterRarity::kGeneralParty:
      stream << "GeneralParty";
      break;
    case MonsterRarity::kChampionParty:
      stream << "ChampionParty";
      break;
    case MonsterRarity::kUniqueParty:
      stream << "UniqueParty";
      break;
    case MonsterRarity::kGiantParty:
      stream << "GiantParty";
      break;
    case MonsterRarity::kTitanParty:
      stream << "TitanParty";
      break;
    case MonsterRarity::kEliteParty:
      stream << "EliteParty";
      break;
    case MonsterRarity::kUnique2Party:
      stream << "Unique2Party";
      break;
    default:
      stream << "UNKNOWN";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream &stream, ItemRarity rarity) {
  switch (rarity) {
    case ItemRarity::kWhite:
      stream << "White";
      break;
    case ItemRarity::kBlue:
      stream << "Blue";
      break;
    case ItemRarity::kSox:
      stream << "Sox";
      break;
    case ItemRarity::kSet:
      stream << "Set";
      break;
    case ItemRarity::kRareSet:
      stream << "RareSet";
      break;
    case ItemRarity::kLegend:
      stream << "Legend";
      break;
    default:
      stream << "UNKNOWN";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream &stream, LifeState lifeState) {
  switch(lifeState) {
    case LifeState::kEmbryo:
      stream << "Embryo";
      break;
    case LifeState::kAlive:
      stream << "Alive";
      break;
    case LifeState::kDead:
      stream << "Dead";
      break;
    case LifeState::kGone:
      stream << "Gone";
      break;
    default:
      stream << "UNKNOWN";
      break;
  }
  return stream;
}

} // namespace sro::entity