#include "skillEngine.hpp"

namespace state {

void SkillEngine::skillCooldownBegin(sro::scalar_types::ReferenceObjectId skillRefId) {
  if (skillsOnCooldown_.find(skillRefId) != skillsOnCooldown_.end()) {
    throw std::runtime_error("Skill cooldown began, but this skill is already on cooldown");
  }
  skillsOnCooldown_.emplace(skillRefId);
}

void SkillEngine::skillCooldownEnded(sro::scalar_types::ReferenceObjectId skillRefId) {
  if (skillsOnCooldown_.find(skillRefId) == skillsOnCooldown_.end()) {
    throw std::runtime_error("Skill cooldown ended, but we weren't tracking this skill's cooldown");
  }
  skillsOnCooldown_.erase(skillRefId);
}

bool SkillEngine::skillIsOnCooldown(sro::scalar_types::ReferenceObjectId skillRefId) const {
  return skillsOnCooldown_.find(skillRefId) != skillsOnCooldown_.end();
}

bool SkillEngine::alreadyTriedToCastSkill(sro::scalar_types::ReferenceObjectId skillRefId) const {
  auto commandIsCastingThisSkill = [&](const packet::structures::ActionCommand &command) {
    return command.commandType == packet::enums::CommandType::kExecute &&
           command.actionType == packet::enums::ActionType::kCast &&
           command.refSkillId == skillRefId;
  };
  for (const auto &pendingCommand : pendingCommandQueue) {
    if (commandIsCastingThisSkill(pendingCommand)) {
      // Skill is already in our pending queue
      return true;
    }
  }
  for (const auto &acceptedCommand : acceptedCommandQueue) {
    if (commandIsCastingThisSkill(acceptedCommand.command)) {
      // Skill is already in our accepted queue
      return true;
    }
  }
  return false;
}

} // namespace state