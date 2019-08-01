#include "skillData.hpp"

#include <string>

namespace pk2::media {

void SkillData::addSkill(Skill &&skill) {
  skills_.emplace(skill.id, skill);
}

bool SkillData::haveSkillWithId(SkillId id) const {
  return (skills_.find(id) != skills_.end());
}

const Skill& SkillData::getSkillById(SkillId id) const {
  auto it = skills_.find(id);
  if (it == skills_.end()) {
    throw std::runtime_error("Trying to get non-existent skill with id "+std::to_string(id));
  }
  return it->second;
}

const SkillData::SkillMap::size_type SkillData::size() const {
  return skills_.size();
}

} // namespace pk2::media