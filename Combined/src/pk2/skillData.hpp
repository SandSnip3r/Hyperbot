#ifndef PK2_MEDIA_SKILL_DATA_HPP
#define PK2_MEDIA_SKILL_DATA_HPP

#include "../../../common/pk2/ref/skill.hpp"

#include <silkroad_lib/scalar_types.h>

#include <unordered_map>

namespace pk2 {

class SkillData {
public:
  using SkillMap = std::unordered_map<ref::SkillId,ref::Skill>;
  void addSkill(ref::Skill &&skill);
  bool haveSkillWithId(sro::scalar_types::ReferenceObjectId id) const;
  const ref::Skill& getSkillById(sro::scalar_types::ReferenceObjectId id) const;
  int32_t getSkillTotalDuration(sro::scalar_types::ReferenceObjectId id) const;
  const SkillMap::size_type size() const;
  sro::scalar_types::ReferenceObjectId getRootSkillRefId(sro::scalar_types::ReferenceObjectId id) const;
private:
  SkillMap skills_;
};

} // namespace pk2

#endif // PK2_MEDIA_SKILL_DATA_HPP