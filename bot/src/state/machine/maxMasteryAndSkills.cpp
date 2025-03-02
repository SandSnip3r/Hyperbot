#include "maxMasteryAndSkills.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentSkillLearnRequest.hpp"
#include "packet/building/clientAgentSkillMasteryLearnRequest.hpp"

#include <absl/algorithm/container.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <algorithm>

namespace state::machine {

MaxMasteryAndSkills::MaxMasteryAndSkills(Bot &bot, sro::pk2::ref::MasteryId id) : StateMachine(bot), masteryId_(id) {
  stateMachineCreated(kName);
  pushBlockedOpcode(packet::Opcode::kClientAgentSkillMasteryLearnRequest);
  pushBlockedOpcode(packet::Opcode::kClientAgentSkillLearnRequest);
  VLOG(1) << "MaxMasteryAndSkills created for mastery ID " << masteryId_;
}

MaxMasteryAndSkills::~MaxMasteryAndSkills() {
  stateMachineDestroyed();
}

Status MaxMasteryAndSkills::onUpdate(const event::Event *event) {
  std::shared_ptr<entity::Self> selfEntity = bot_.selfState();
  if (event != nullptr) {
    if (auto *leveledUpMastery = dynamic_cast<const event::LearnMasterySuccess*>(event)) {
      if (leveledUpMastery->masteryId == masteryId_) {
        VLOG(1) << absl::StreamFormat("Mastery is now level %d", selfEntity->getMasteryLevel(masteryId_));
        resetTimeout();
      }
    } else if (auto *leveledUpSkill = dynamic_cast<const event::LearnSkillSuccess*>(event)) {
      if (currentLearningSkill_) {
        if (*currentLearningSkill_ == leveledUpSkill->newSkillRefId) {
          VLOG(1) << "Successfully learned " << *currentLearningSkill_;
          currentLearningSkill_.reset();
          resetTimeout();
        } else {
          LOG(WARNING) << "Learned a skill different from the one we were trying to learn. Trying to learn " << *currentLearningSkill_ << " but learned " << leveledUpSkill->newSkillRefId;
        }
      } else {
        LOG(WARNING) << "Learned a skill, but we weren't trying to learn one";
      }
    } else if (event->eventCode == event::EventCode::kTimeout) {
      if (timeoutEventId_ && event->eventId == *timeoutEventId_) {
        VLOG(2) << "Timed out!";
        timeoutEventId_.reset();
      }
    } else if (event->eventCode == event::EventCode::kLearnSkillError) {
      LOG(WARNING) << "Error learning skill. Aborting state machine";
      return Status::kDone;
    }
  }

  if (timeoutEventId_.has_value()) {
    VLOG(3) << "Still waiting on a response to the last packet we sent";
    return Status::kNotDone;
  }

  // First, max mastery level.
  const uint8_t currentMasteryLevel = selfEntity->getMasteryLevel(masteryId_);
  const uint8_t currentLevel = selfEntity->getCurrentLevel();
  if (currentMasteryLevel < currentLevel) {
    // Still need to level up mastery.
    VLOG(1) << absl::StreamFormat("Mastery is level %d, character level is %d. Sending packet to level up mastery", currentMasteryLevel, currentLevel);
    bot_.packetBroker().injectPacket(packet::building::ClientAgentSkillMasteryLearnRequest::incrementLevel(masteryId_), PacketContainer::Direction::kBotToServer);
    timeoutEventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(200));
    return Status::kNotDone;
  }

  if (!skillTree_.initialized()) {
    VLOG(1) << "Mastery matches current level. Leveling up skills";
    const std::vector<packet::structures::Skill> ourSkills = selfEntity->skills();
    std::vector<sro::scalar_types::ReferenceSkillId> ourSkillIds;
    for (const packet::structures::Skill &skill : ourSkills) {
      ourSkillIds.push_back(skill.id);
    }
    skillTree_.initialize(bot_.gameData().skillData(), masteryId_, currentMasteryLevel, ourSkillIds);
  }

  if (!skillTree_.haveSkillToLearn()) {
    VLOG(1) << "No more skills to learn!";
    return Status::kDone;
  }

  sro::scalar_types::ReferenceSkillId nextSkillId;
  if (currentLearningSkill_) {
    nextSkillId = *currentLearningSkill_;
  } else {
    nextSkillId = skillTree_.getNextSkillToLearn();
  }
  VLOG(1) << "Sending packet to learn skill " << nextSkillId;
  bot_.packetBroker().injectPacket(packet::building::ClientAgentSkillLearnRequest::learnSkill(nextSkillId), PacketContainer::Direction::kBotToServer);
  currentLearningSkill_ = nextSkillId;
  timeoutEventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(400));
  return Status::kNotDone;
}

void MaxMasteryAndSkills::resetTimeout() {
  if (timeoutEventId_) {
    bool success = bot_.eventBroker().cancelDelayedEvent(*timeoutEventId_);
    if (!success) {
      LOG(WARNING) << "Hmm, cancelling delayed event failed";
    }
    timeoutEventId_.reset();
  } else {
    LOG(WARNING) << "Do not have a timeout event!";
  }
}

namespace internal {

void SkillTree::initialize(const pk2::SkillData &skillData, sro::pk2::ref::MasteryId masteryId, uint8_t masteryLevel, const std::vector<sro::scalar_types::ReferenceSkillId> &knownSkillIds) {
  // Get all skills for this mastery.
  auto skills = skillData.getSkillIdsForMastery(masteryId);
  VLOG(3) << absl::StreamFormat("All skills for mastery: [%s]", absl::StrJoin(skills, ","));

  // Gather a list of all skills which are part of a chain and not the first in the chain.
  absl::flat_hash_set<sro::pk2::ref::SkillId> nonRootChainSkills;
  for (const sro::pk2::ref::SkillId skillId : skills) {
    const sro::pk2::ref::Skill &skill = skillData.getSkillById(skillId);
    if (skill.basicChainCode != 0) {
      nonRootChainSkills.insert(skill.basicChainCode);
    }
  }

  // Remove some skills.
  skills.erase(std::remove_if(skills.begin(), skills.end(), [&](const auto skillId) {
    const auto &skill = skillData.getSkillById(skillId);
    if (skill.reqCommonMasteryLevel1 > masteryLevel) {
      // Remove skills which are higher level than our mastery currently is.
      VLOG(3) << "Removing skill " << skillId << " because it is higher level than our mastery currently is";
      return true;
    }
    if (skill.reqCommonMasteryLevel1 == 0) {
      // Remove common attacks.
      VLOG(3) << "Removing skill " << skillId << " because it is a common attack";
      return true;
    }
    if (absl::c_linear_search(nonRootChainSkills, skillId)) {
      // Remove skills which are parts of a chain and not the first in the chain.
      VLOG(3) << "Removing skill " << skillId << " because this skill is a part of a chain and not the first in the chain";
      return true;
    }
    return false;
  }), skills.end());
  // NOTE: At this point, skills which we already know are still in this list.

  VLOG(3) << absl::StreamFormat("Cleaned up skill list: [%s]", absl::StrJoin(skills, ","));

  // Create groups of skills (using GroupId).
  // Create a dependency map between groups; some groups must be completed before others.
  absl::flat_hash_map<sro::pk2::ref::Skill::GroupId, std::vector<sro::pk2::ref::SkillId>> groupToSkillsMap;
  absl::flat_hash_map<sro::pk2::ref::Skill::GroupId, absl::flat_hash_set<sro::pk2::ref::Skill::GroupId>> groupDependencyMap;
  for (const auto skillId : skills) {
    const auto &skill = skillData.getSkillById(skillId);
    groupToSkillsMap[skill.groupId].push_back(skillId);
    if (skill.reqLearnSkill1 != 0) {
      groupDependencyMap[skill.groupId].insert(skill.reqLearnSkill1);
    }
    if (skill.reqLearnSkill2 != 0) {
      groupDependencyMap[skill.groupId].insert(skill.reqLearnSkill2);
    }
    if (skill.reqLearnSkill3 != 0) {
      groupDependencyMap[skill.groupId].insert(skill.reqLearnSkill3);
    }
  }

  // Sort skills by level in their groups.
  for (auto &groupSkillsPair : groupToSkillsMap) {
    std::sort(groupSkillsPair.second.begin(), groupSkillsPair.second.end(), [&](const sro::pk2::ref::SkillId lhsSkillId, const sro::pk2::ref::SkillId rhsSkillId) {
      const sro::pk2::ref::Skill &lhsSkill = skillData.getSkillById(lhsSkillId);
      const sro::pk2::ref::Skill &rhsSkill = skillData.getSkillById(rhsSkillId);
      return lhsSkill.reqCommonMasteryLevel1 < rhsSkill.reqCommonMasteryLevel1;
    });
  }

  // Remove skills which we already know. We do this separately, because when we remove a skill, we also need to remove every skill lower level than it, within its group.
  for (auto &groupSkillsPair : groupToSkillsMap) {
    std::vector<sro::pk2::ref::SkillId> &skillIds = groupSkillsPair.second;
    for (int i=skillIds.size()-1; i>=0; --i) {
      if (absl::c_linear_search(knownSkillIds, skillIds.at(i))) {
        // Already know this skill. Delete everything before it.
        skillIds.erase(skillIds.begin(), skillIds.begin()+i+1);
        break;
      }
    }
  }

  VLOG(3) << absl::StreamFormat("Group to sorted skills map: {\n  %s\n}", absl::StrJoin(groupToSkillsMap, "\n  ", [](std::string *out, const auto &groupIdSetPair) {
    absl::StrAppend(out, absl::StrFormat("%d: [%s]", groupIdSetPair.first, absl::StrJoin(groupIdSetPair.second, ",")));
  }));
  VLOG(3) << absl::StreamFormat("Group dependency map: {\n  %s\n}", absl::StrJoin(groupDependencyMap, "\n  ", [](std::string *out, const auto &groupIdSetPair) {
    absl::StrAppend(out, absl::StrFormat("%d: [%s]", groupIdSetPair.first, absl::StrJoin(groupIdSetPair.second, ",")));
  }));

  // Create a group ordering.
  // Things which have no entry in `groupDependencyMap` depend on nothing.
  // Things which are not depended on are the last to level up (root).
  std::vector<sro::pk2::ref::Skill::GroupId> groupOrdering;
  std::set<sro::pk2::ref::Skill::GroupId> allGroups; // Must be an ordered container for c_set_difference.
  for (const auto& [groupId, skills] : groupToSkillsMap) {
    allGroups.insert(groupId);
  }
  while (!allGroups.empty()) {
    std::set<sro::pk2::ref::Skill::GroupId> groupsThatDependOnSomething; // Must be an ordered container for c_set_difference.
    for (const auto &groupAndSetPair : groupDependencyMap) {
      groupsThatDependOnSomething.insert(groupAndSetPair.first);
    }
    absl::flat_hash_set<sro::pk2::ref::Skill::GroupId> groupsThatDependOnNothing;
    absl::c_set_difference(allGroups, groupsThatDependOnSomething, std::inserter(groupsThatDependOnNothing, groupsThatDependOnNothing.end()));
    for (const auto groupId : groupsThatDependOnNothing) {
      groupOrdering.push_back(groupId);
      // If any group depends on this one, remove it.
      // If it's their last dependency, delete the entire entry.
      for (auto &groupIdDepPair : groupDependencyMap) {
        groupIdDepPair.second.erase(groupId);
      }
      for (auto it=groupDependencyMap.begin(); it!=groupDependencyMap.end();) {
        if (it->second.empty()) {
          groupDependencyMap.erase(it++);
        } else {
          ++it;
        }
      }
    }
    allGroups = groupsThatDependOnSomething;
  }
  VLOG(3) << absl::StreamFormat("Final group ordering: [%s]", absl::StrJoin(groupOrdering, ","));

  for (const auto groupId : groupOrdering) {
    const auto &skillsInGroup = groupToSkillsMap.at(groupId);
    skillOrdering_.insert(skillOrdering_.end(), skillsInGroup.begin(), skillsInGroup.end());
  }
  VLOG(3) << absl::StreamFormat("Final skill ordering: [%s]", absl::StrJoin(skillOrdering_, ","));
  absl::c_reverse(skillOrdering_);

  initialzed_ = true;
}

sro::scalar_types::ReferenceSkillId SkillTree::getNextSkillToLearn() {
  if (skillOrdering_.empty()) {
    throw std::runtime_error("No next skill to learn");
  }
  const auto skill = skillOrdering_.back();
  skillOrdering_.pop_back();
  return skill;
}

} // namespace internal

} // namespace state::machine
