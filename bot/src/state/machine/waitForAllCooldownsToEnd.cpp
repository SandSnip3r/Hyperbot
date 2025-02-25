#include "waitForAllCooldownsToEnd.hpp"

#include "bot.hpp"

#include <absl/log/log.h>

namespace state::machine {

WaitForAllCooldownsToEnd::WaitForAllCooldownsToEnd(Bot &bot) : StateMachine(bot) {
}

WaitForAllCooldownsToEnd::~WaitForAllCooldownsToEnd() {
}

Status WaitForAllCooldownsToEnd::onUpdate(const event::Event *event) {
  const std::map<type_id::TypeId, broker::EventBroker::EventId> &itemCooldownEventIds = bot_.selfState()->getItemCooldownEventIdMap();
  const absl::flat_hash_map<sro::scalar_types::ReferenceObjectId, broker::EventBroker::EventId> &skillCooldownEventIds = bot_.selfState()->skillEngine.getSkillCooldownEventIdMap();
  if (itemCooldownEventIds.empty() && skillCooldownEventIds.empty()) {
    VLOG(1) << characterNameForLog() << "No items or skills on cooldown";
    return Status::kDone;
  }
  for (const auto &[itemTypeId, eventId] : itemCooldownEventIds) {
    std::optional<std::chrono::milliseconds> remainingTime = bot_.eventBroker().timeRemainingOnDelayedEvent(eventId);
    if (!remainingTime) {
      throw std::runtime_error("Have an event ID for item cooldown, but no time left");
    }
    VLOG(1) << characterNameForLog() << "Event ID " << eventId << ", item with typeId " << type_id::toString(itemTypeId) << " has " << remainingTime->count() << "ms left";
  }
  for (const auto &[skillId, eventId] : skillCooldownEventIds) {
    std::optional<std::chrono::milliseconds> remainingTime = bot_.eventBroker().timeRemainingOnDelayedEvent(eventId);
    if (!remainingTime) {
      throw std::runtime_error("Have an event ID for skill cooldown, but no time left");
    }
    VLOG(1) << characterNameForLog() << "Event ID " << eventId << ", skill " << bot_.gameData().getSkillName(skillId) << " has " << remainingTime->count() << "ms left";
  }
  return Status::kNotDone;
}

} // namespace state::machine
