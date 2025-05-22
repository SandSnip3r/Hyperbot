#ifndef PK2_REF_MASTERY_HPP_
#define PK2_REF_MASTERY_HPP_

#include <cstdint>
#include <string>
#include <ostream>

namespace sro::pk2::ref {

using MasteryId = int32_t;

struct Mastery {
  // Mastery ID
  MasteryId masteryId;
  // Mastery Name - Do Not Use
  std::string masteryName;
  // MasteryNameCode
  std::string masteryNameCode;
  // GroupNum
  int32_t groupNum;
  // Mastery Description ID
  std::string masteryDescriptionId;
  // Tab Name Code
  std::string tabNameCode;
  // Type (TabID)
  int tabId;
  // "SkillToolTipType...
  //  0: General weapon skills
  //  1: General Technique
  //  2: Energy and blood vs. law.. (a skill that does not affect those skills even if the level increases)
  int skillToolTipType;
  // Weapon Type 1
  int weaponType1;
  // Weapon Type 2
  int weaponType2;
  // Weapon Type 3
  int weaponType3;
  // Mastery Icon
  std::string masteryIcon;
  // Mastery Focus Icon
  std::string masteryFocusIcon;
};

} // namespace sro::pk2::ref

#endif // PK2_REF_MASTERY_HPP_