#include "skillData.hpp"

#include <string>

namespace pk2 {

void SkillData::addSkill(ref::Skill &&skill) {
  skills_.emplace(skill.id, skill);
}

bool SkillData::haveSkillWithId(ref::SkillId id) const {
  return (skills_.find(id) != skills_.end());
}

const ref::Skill& SkillData::getSkillById(ref::SkillId id) const {
  auto it = skills_.find(id);
  if (it == skills_.end()) {
    throw std::runtime_error("Trying to get non-existent skill with id "+std::to_string(id));
  }
  return it->second;
}

const SkillData::SkillMap::size_type SkillData::size() const {
  return skills_.size();
}

} // namespace pk2