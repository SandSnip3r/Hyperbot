#ifndef PK2_MEDIA_SKILL_HPP_
#define PK2_MEDIA_SKILL_HPP_

#include <string>

namespace pk2::media {

using SkillId = uint32_t;

struct Skill {
	SkillId id;
	std::string basicCode;
  std::string basicName;
  std::string basicGroup;
};

} // namespace pk2::media

#endif // PK2_MEDIA_SKILL_HPP_