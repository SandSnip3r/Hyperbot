#include "bot.hpp"
#include "event/event.hpp"
#include "rl/observation.hpp"

#include <absl/algorithm/container.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <array>

namespace rl {

namespace {

int countItemsInInventory(const Bot &bot, sro::scalar_types::ReferenceObjectId itemId) {
  int count=0;
  for (const auto &item : bot.selfState()->inventory) {
    if (item.refItemId == itemId) {
      count += item.getQuantity();
    }
  }
  return count;
}
} // namespace

Observation::Observation(const Bot &bot, const event::Event *event, sro::scalar_types::EntityGlobalId opponentGlobalId) {
  eventCode_ = event->eventCode;
  if (!bot.selfState()) {
    throw std::runtime_error("Cannot get an observation without a self state");
  }
  ourCurrentHp_ = bot.selfState()->currentHp();
  ourMaxHp_ = bot.selfState()->maxHp().value();
  ourCurrentMp_ = bot.selfState()->currentMp();
  ourMaxMp_ = bot.selfState()->maxMp().value();

  std::shared_ptr<entity::Self> opponent = bot.entityTracker().getEntity<entity::Self>(opponentGlobalId);
  opponentCurrentHp_ = opponent->currentHp();
  opponentMaxHp_ = opponent->maxHp().value();
  opponentCurrentMp_ = opponent->currentMp();
  opponentMaxMp_ = opponent->maxMp().value();

  hpPotionCount_ = countItemsInInventory(bot, 5);

  static constexpr std::array kSkills{28, 131, 554, 1253, 1256, 1271, 1272, 1281, 1335, 1377, 1380, 1398, 1399, 1410, 1421, 1441, 8312, 21209, 30577, 37, 114, 298, 300, 322, 339, 371, 588, 610, 644, 1315, 1343, 1449};
  for (sro::scalar_types::ReferenceSkillId skillId : kSkills) {
    std::optional<std::chrono::milliseconds> remainingCooldown = bot.selfState()->skillEngine.skillRemainingCooldown(skillId, bot.eventBroker());
    skillCooldowns_.emplace_back(remainingCooldown.value_or(std::chrono::milliseconds(0)).count());
  }

  static constexpr std::array kItems{5, 12, 56};
  const std::map<type_id::TypeId, broker::EventBroker::EventId> &itemCooldownEventIds = bot.selfState()->getItemCooldownEventIdMap();
  for (sro::scalar_types::ReferenceObjectId itemId : kItems) {
    const auto it = itemCooldownEventIds.find(itemId);
    if (it == itemCooldownEventIds.end()) {
      itemCooldowns_.emplace_back(0);
    } else {
      std::optional<std::chrono::milliseconds> remainingCooldown = bot.eventBroker().timeRemainingOnDelayedEvent(it->second);
      itemCooldowns_.emplace_back(remainingCooldown.value_or(std::chrono::milliseconds(0)).count());
    }
  }
}

std::string Observation::toString() const {
  return absl::StrFormat("{hp:%d/%d, mp:%d/%d, opponentHp:%d/%d, opponentMp:%d/%d, skillCooldowns:[%s], itemCooldowns:[%s]}", ourCurrentHp_,  ourMaxHp_, ourCurrentMp_,  ourMaxMp_, opponentCurrentHp_,  opponentMaxHp_, opponentCurrentMp_,  opponentMaxMp_, absl::StrJoin(skillCooldowns_, ", "), absl::StrJoin(itemCooldowns_, ", "));
}

}