#ifndef PK2_MEDIA_SKILL_DATA_HPP
#define PK2_MEDIA_SKILL_DATA_HPP

#include <silkroad_lib/pk2/ref/skill.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <unordered_map>

namespace pk2 {

class SkillData {
private:
  using SkillMap = std::unordered_map<sro::pk2::ref::SkillId, sro::pk2::ref::Skill>;
public:
  using SizeType = SkillMap::size_type;
  void addSkill(sro::pk2::ref::Skill &&skill);
  bool haveSkillWithId(sro::scalar_types::ReferenceSkillId id) const;

  // Throws if we have no such skill.
  const sro::pk2::ref::Skill& getSkillById(sro::scalar_types::ReferenceSkillId id) const;
  int32_t getSkillTotalDuration(sro::scalar_types::ReferenceSkillId id) const;
  const SizeType size() const;
  sro::scalar_types::ReferenceSkillId getRootSkillRefId(sro::scalar_types::ReferenceSkillId id) const;

  std::vector<sro::pk2::ref::SkillId> getSkillIdsForMastery(sro::scalar_types::ReferenceMasteryId masteryId) const;
private:
  SkillMap skills_;
};

} // namespace pk2

#endif // PK2_MEDIA_SKILL_DATA_HPP