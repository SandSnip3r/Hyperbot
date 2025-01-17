#include "skillData.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace pk2 {

void SkillData::addSkill(sro::pk2::ref::Skill &&skill) {
  skills_.emplace(skill.id, skill);
}

bool SkillData::haveSkillWithId(sro::scalar_types::ReferenceObjectId id) const {
  return (skills_.find(id) != skills_.end());
}

const sro::pk2::ref::Skill& SkillData::getSkillById(sro::scalar_types::ReferenceObjectId id) const {
  auto it = skills_.find(id);
  if (it == skills_.end()) {
    throw std::runtime_error("Trying to get non-existent skill with id "+std::to_string(id));
  }
  return it->second;
}

int32_t SkillData::getSkillTotalDuration(sro::scalar_types::ReferenceObjectId id) const {
  int32_t sum = 0;
  auto skill = getSkillById(id);
  sum += skill.actionPreparingTime + skill.actionCastingTime + skill.actionActionDuration;
  while (skill.basicChainCode != 0) {
    skill = getSkillById(skill.basicChainCode);
    sum += skill.actionPreparingTime + skill.actionCastingTime + skill.actionActionDuration;
  }
  return sum;
}

const SkillData::SkillMap::size_type SkillData::size() const {
  return skills_.size();
}

sro::scalar_types::ReferenceObjectId SkillData::getRootSkillRefId(sro::scalar_types::ReferenceObjectId id) const {
  // basicChainCode tells us which skill in a chain comes next, there is no equivalent for going backwards. We need to search over all skills to find if another skill references this one as the next skill in the chain.
  // TODO: (Optimization) Precompute the reverse chain codes
  // Try to find a skill which has this skill's id as its basic chain code
  bool foundRoot{false};
  while (!foundRoot) {
    bool foundParent{false};
    for (const auto &i : skills_) {
      if (i.second.basicChainCode == id) {
        // Found parent
        id = i.first;
        foundParent = true;
        break;
      }
    }
    if (!foundParent) {
      foundRoot = true;
    }
  }
  return id;
}

std::vector<sro::pk2::ref::SkillId> SkillData::getSkillIdsForMastery(sro::scalar_types::ReferenceMasteryId masteryId) const {
  std::vector<sro::pk2::ref::SkillId> result;
  for (auto &skillIdSkillPair : skills_) {
    if (skillIdSkillPair.second.reqCommonMastery1 == masteryId ||
        skillIdSkillPair.second.reqCommonMastery2 == masteryId) {
      result.push_back(skillIdSkillPair.first);
    }
  }
  return result;
}

} // namespace pk2