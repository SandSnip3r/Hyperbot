#ifndef PK2_MEDIA_SKILL_HPP_
#define PK2_MEDIA_SKILL_HPP_

#include <array>
#include <string>

namespace pk2::ref {

using SkillId = uint32_t;

struct Skill {
	SkillId id;
	std::string basicCode;
  std::string basicName;
  std::string basicGroup;
  std::array<int32_t,50> params;
  
  // Efta or atfe ("auto transfer effect") like Recovery Division or bard dances
  bool isEfta() const;
};

} // namespace pk2::ref

#endif // PK2_MEDIA_SKILL_HPP_