#include "skillEngine.hpp"

#include <absl/log/log.h>

namespace state {

void SkillEngine::skillCooldownBegin(sro::scalar_types::ReferenceObjectId skillRefId, broker::EventBroker::EventId cooldownEndEventId) {
  if (auto it = skillCooldownEventIdMap_.find(skillRefId); it != skillCooldownEventIdMap_.end()) {
    // Skill is already on cooldown. We must trust that the user cancelled the old cooldown ended event and triggered a new one.
    // We will overwrite the old one with the newly given one.
    it->second = cooldownEndEventId;
    return;
  }
  // Skill is not already on cooldown.
  skillCooldownEventIdMap_[skillRefId] = cooldownEndEventId;
}

void SkillEngine::skillCooldownEnded(sro::scalar_types::ReferenceObjectId skillRefId) {
  const auto it = skillCooldownEventIdMap_.find(skillRefId);
  if (it == skillCooldownEventIdMap_.end()) {
    // Skill cooldown ended, but we weren't tracking this skill's cooldown.
    LOG(WARNING) << "Skill cooldown ended, but we weren't tracking this skill's cooldown.";
    return;
  }
  skillCooldownEventIdMap_.erase(it);
}

bool SkillEngine::skillIsOnCooldown(sro::scalar_types::ReferenceObjectId skillRefId) const {
  return skillCooldownEventIdMap_.find(skillRefId) != skillCooldownEventIdMap_.end();
}

std::optional<broker::EventBroker::EventId> SkillEngine::getSkillCooldownEndEventId(sro::scalar_types::ReferenceObjectId skillRefId) const {
  const auto it = skillCooldownEventIdMap_.find(skillRefId);
  if (it == skillCooldownEventIdMap_.end()) {
    return {};
  }
  return it->second;
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

void SkillEngine::reset() {
  skillCastIdMap.clear();
  pendingCommandQueue.clear();
  acceptedCommandQueue.clear();
}

void SkillEngine::cancelEvents(broker::EventBroker &eventBroker) {
  for (const auto refEventPair : skillCooldownEventIdMap_) {
    eventBroker.cancelDelayedEvent(refEventPair.second);
  }
}

} // namespace state