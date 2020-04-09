#ifndef PK2_MEDIA_SKILL_HPP_
#define PK2_MEDIA_SKILL_HPP_

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace pk2::ref {

using SkillId = int32_t;

struct Skill {
	SkillId id;
	std::string basicCode;
  std::string basicName;
  std::string basicGroup;
  int32_t actionReuseDelay;
  bool targetRequired;
  uint8_t reqCastWeapon1;
  uint8_t reqCastWeapon2;
  std::array<int32_t,50> params;
  
  // Efta or atfe ("auto transfer effect") like Recovery Division or bard dances
  bool isEfta() const;
  std::vector<std::pair<uint8_t, uint8_t>> reqi() const;
};

} // namespace pk2::ref

#endif // PK2_MEDIA_SKILL_HPP_