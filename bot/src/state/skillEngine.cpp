#include "skillEngine.hpp"

#include <absl/log/log.h>

namespace state {

void SkillEngine::skillCooldownBegin(sro::scalar_types::ReferenceObjectId skillRefId, broker::EventBroker::EventId cooldownEndEventId) {
  if (skillCooldownEventIdMap_.find(skillRefId) != skillCooldownEventIdMap_.end()) {
    throw std::runtime_error(absl::StrFormat("Skill %d cooldown began, but this skill is already on cooldown", skillRefId));
  }
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

std::optional<std::chrono::milliseconds> SkillEngine::skillRemainingCooldown(sro::scalar_types::ReferenceObjectId skillRefId, const broker::EventBroker &eventBroker) const {
  const auto it = skillCooldownEventIdMap_.find(skillRefId);
  if (it == skillCooldownEventIdMap_.end()) {
    return {};
  }
  return eventBroker.timeRemainingOnDelayedEvent(it->second);
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