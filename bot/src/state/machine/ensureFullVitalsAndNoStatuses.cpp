#include "ensureFullVitalsAndNoStatuses.hpp"

#include "bot.hpp"
#include "state/machine/gmCommandSpawnAndPickItems.hpp"
#include "state/machine/useItem.hpp"
#include "type_id/categories.hpp"

#include <absl/log/log.h>

namespace state::machine {

EnsureFullVitalsAndNoStatuses::EnsureFullVitalsAndNoStatuses(Bot &bot) : StateMachine(bot) {
  vigorPotionItemId_ = bot_.gameData().itemData().getItemId([](const sro::pk2::ref::Item &item) {
    return type_id::categories::kVigorPotion.contains(type_id::getTypeId(item)) && item.itemClass == 5;
  });
  universalPillItemId_ = bot_.gameData().itemData().getItemId([](const sro::pk2::ref::Item &item) {
    return type_id::categories::kUniversalPill.contains(type_id::getTypeId(item)) && item.itemClass == 5;
  });
  purificationPillItemId_ = bot_.gameData().itemData().getItemId([](const sro::pk2::ref::Item &item) {
    return type_id::categories::kPurificationPill.contains(type_id::getTypeId(item)) && item.itemClass == 4;
  });
}

EnsureFullVitalsAndNoStatuses::~EnsureFullVitalsAndNoStatuses() {
}

Status EnsureFullVitalsAndNoStatuses::onUpdate(const event::Event *event) {
  if (event != nullptr) {
    // Look for a potion use.
    if (const auto *itemUseSuccess = dynamic_cast<const event::ItemUseSuccess*>(event); itemUseSuccess != nullptr) {
      if (itemUseSuccess->globalId == bot_.selfState()->globalId && itemUseSuccess->refId == vigorPotionItemId_) {
        CHAR_VLOG(1) << "Used a health potion, need to wait at least 5s until we can say that we're done";
        if (waitForPotionEventId_) {
          CHAR_VLOG(1) << "Cancelling previous wait for potion event";
          bot_.eventBroker().cancelDelayedEvent(*waitForPotionEventId_);
        }
        // TODO: If we move this out to PvpManager, we can overlap some work.
        waitForPotionEventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::seconds(5));
      }
    } else if (event->eventCode == event::EventCode::kTimeout &&
               waitForPotionEventId_ &&
               *waitForPotionEventId_ == event->eventId) {
      CHAR_VLOG(1) << "Wait for potion event is done";
      waitForPotionEventId_.reset();
    }
  }

  if (childState_ != nullptr) {
    // If we're in the middle of something, keep doing it.
    const Status status = childState_->onUpdate(event);
    if (status == Status::kNotDone) {
      return status;
    }
    CHAR_VLOG(1) << "Child state machine is done";
    childState_.reset();
  }

  CHAR_VLOG(1) << "No child state machine. Checking vitals";
  std::vector<common::ItemRequirement> fullVitalsItemRequirements;
  const bool hpNotFull = bot_.selfState()->currentHp() < bot_.selfState()->maxHp();
  const bool mpNotFull = bot_.selfState()->currentMp() < bot_.selfState()->maxMp();
  const bool spawnAndUseVigor = !waitForPotionEventId_.has_value() && (hpNotFull || mpNotFull);
  if (spawnAndUseVigor) {
    // Spawn and use a vigor potion
    fullVitalsItemRequirements.push_back(common::ItemRequirement{vigorPotionItemId_, 1});
  }

  bool haveLegacyState = false;
  for (uint16_t legacyStateEffect : bot_.selfState()->legacyStateEffects()) {
    if (legacyStateEffect != 0) {
      CHAR_VLOG(1) << "Has a legacy state";
      haveLegacyState = true;
      break;
    }
  }
  if (haveLegacyState) {
    fullVitalsItemRequirements.push_back(common::ItemRequirement{universalPillItemId_, 1});
  }

  bool haveModernState = false;
  for (uint8_t modernStateLevel : bot_.selfState()->modernStateLevels()) {
    if (modernStateLevel != 0) {
      CHAR_VLOG(1) << "Has a modern state";
      haveModernState = true;
      break;
    }
  }
  if (haveModernState) {
    fullVitalsItemRequirements.push_back(common::ItemRequirement{purificationPillItemId_, 1});
  }

  if (fullVitalsItemRequirements.empty()) {
    CHAR_VLOG(1) << "Vitals are full and there are no statuses";
    if (waitForPotionEventId_) {
      CHAR_VLOG(1) << "Need to wait for potion to stop healing";
      return Status::kNotDone;
    }
    return Status::kDone;
  }

  CHAR_VLOG(1) << "Vitals are not full or there are statuses. Spawning, picking, and using " << fullVitalsItemRequirements.size() << " items";
  setChildStateMachine<state::machine::SequentialStateMachines>();
  state::machine::SequentialStateMachines &sequentialStateMachines = dynamic_cast<state::machine::SequentialStateMachines&>(*childState_);
  sequentialStateMachines.emplace<state::machine::GmCommandSpawnAndPickItems>(fullVitalsItemRequirements);
  if (spawnAndUseVigor) {
    sequentialStateMachines.emplace<state::machine::UseItem>(vigorPotionItemId_);
  }
  if (haveLegacyState) {
    sequentialStateMachines.emplace<state::machine::UseItem>(universalPillItemId_);
  }
  if (haveModernState) {
    sequentialStateMachines.emplace<state::machine::UseItem>(purificationPillItemId_);
  }
  // TODO: It is not enough to simply use the item to know that our vitals are good. Shortly after using the item, we expect a vitals-changed event. If we evaluate our vitals after we used the item but before this update comes, we'll falsely assume our vitals are still bad.
  return childState_->onUpdate(event);
}

} // namespace state::machine
