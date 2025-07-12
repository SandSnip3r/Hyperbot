#include "bot.hpp"
#include "rl/observation.hpp"
#include "rl/observationBuilder.hpp"

namespace rl {

namespace {

uint16_t countItemsInInventory(const Bot &bot, sro::scalar_types::ReferenceObjectId itemId) {
  uint16_t count=0;
  for (const auto &item : bot.selfState()->inventory) {
    if (item.refItemId == itemId) {
      count += item.getQuantity();
    }
  }
  return count;
}

} // namespace

void ObservationBuilder::buildObservationFromBot(const Bot &bot,
                             Observation &observation,
                             sro::scalar_types::EntityGlobalId opponentGlobalId) {
  // =================================================== Skills ===================================================
  for (size_t i=0; i<kSkillIdsForObservations.size(); ++i) {
    const sro::scalar_types::ReferenceSkillId skillId = kSkillIdsForObservations[i];
    const std::optional<std::chrono::milliseconds> remainingCooldown = bot.selfState()->skillRemainingCooldown(skillId);
    const bool isAvailable = !remainingCooldown.has_value() || *remainingCooldown == std::chrono::milliseconds(0);
    observation.skillData_[i].setSkillIsAvailable(skillId, isAvailable);
  }
  // =================================================== Items ====================================================
  const std::map<type_id::TypeId, broker::EventBroker::EventId> &itemCooldownEventIds = bot.selfState()->getItemCooldownEventIdMap();
  for (size_t i=0; i<kItemIdsForObservations.size(); ++i) {
    const sro::scalar_types::ReferenceObjectId itemId = kItemIdsForObservations[i];
    const auto it = itemCooldownEventIds.find(itemId);
    const bool isOnCooldown = [&]() {
      if (it == itemCooldownEventIds.end()) {
        return false;
      } else {
        const std::optional<std::chrono::milliseconds> remainingCooldown = bot.eventBroker().timeRemainingOnDelayedEvent(it->second);
        if (!remainingCooldown) {
          LOG(WARNING) << "Weird that there is a cooldown event ID but the event broker's time remaining returned an empty optional";
          return false;
        } else {
          return *remainingCooldown != std::chrono::milliseconds(0);
        }
      }
    }();
    uint16_t countAvailable = countItemsInInventory(bot, itemId);
    constexpr uint16_t maxCount = 5; // IF-CHANGE: If we change this, also change TrainingManager::buildItemRequirementList
    observation.itemData_[i].setItemOnCooldownAndCount(itemId, isOnCooldown, countAvailable, maxCount);
  }
  // =================================================== HP/MP ====================================================
  observation.ourHpData_.setCurrentAndMax(bot.selfState()->currentHp(), bot.selfState()->maxHp().value());
  observation.ourMpData_.setCurrentAndMax(bot.selfState()->currentMp(), bot.selfState()->maxMp().value());

  std::shared_ptr<entity::Self> opponent = bot.entityTracker().getEntity<entity::Self>(opponentGlobalId);
  if (!opponent) {
    throw std::runtime_error(absl::StrFormat("Cannot find opponent with global id %d when creating observation", opponentGlobalId));
  }
  observation.opponentHpData_.setCurrentAndMax(opponent->currentHp(), opponent->maxHp().value());
  // ==============================================================================================================
}

} // namespace rl