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
  timestamp_ = std::chrono::high_resolution_clock::now();
  eventCode_ = event->eventCode;
  if (!bot.selfState()) {
    throw std::runtime_error("Cannot get an observation without a self state");
  }
  ourCurrentHp_ = bot.selfState()->currentHp();
  ourMaxHp_ = bot.selfState()->maxHp().value();
  ourCurrentMp_ = bot.selfState()->currentMp();
  ourMaxMp_ = bot.selfState()->maxMp().value();
  weAreKnockedDown_ = bot.selfState()->stunnedFromKnockdown;

  std::shared_ptr<entity::Self> opponent = bot.entityTracker().getEntity<entity::Self>(opponentGlobalId);
  if (!opponent) {
    throw std::runtime_error(absl::StrFormat("Cannot find opponent with global id %d when creating observation", opponentGlobalId));
  }
  opponentCurrentHp_ = opponent->currentHp();
  opponentMaxHp_ = opponent->maxHp().value();
  opponentCurrentMp_ = opponent->currentMp();
  opponentMaxMp_ = opponent->maxMp().value();
  opponentIsKnockedDown_ = opponent->stunnedFromKnockdown;

  hpPotionCount_ = countItemsInInventory(bot, 5);

  static constexpr std::array kBuffs{1441, 131, 1272, 1421, 30577, 1399, 28, 554, 1410, 1398, 1335, 1271, 1380, 1256, 1253, 1377, 1281};
  if (kBuffs.size() != remainingTimeOurBuffs_.size() ||
      kBuffs.size() != remainingTimeOpponentBuffs_.size()) {
    throw std::runtime_error("Local buff array and observation's buff array do not have the same size");
  }
  for (int i=0; i<kBuffs.size(); ++i) {
    const sro::scalar_types::ReferenceSkillId skillId = kBuffs[i];
    // ==================================== Us ====================================
    if (bot.selfState()->buffIsActive(skillId)) {
      std::optional<entity::Character::BuffData::ClockType::time_point> castTime = bot.selfState()->buffCastTime(skillId);
      if (castTime) {
        remainingTimeOurBuffs_[i] = std::chrono::duration_cast<std::chrono::milliseconds>(entity::Character::BuffData::ClockType::now()-*castTime).count();
      } else {
        LOG(WARNING) << "Buff " << bot.gameData().getSkillName(skillId) << " is active for us, but no cast time is known";
        remainingTimeOurBuffs_[i] = 0;
      }
    } else {
      remainingTimeOurBuffs_[i] = 0;
    }
    // ================================= Opponent =================================
    if (opponent->buffIsActive(skillId)) {
      std::optional<entity::Character::BuffData::ClockType::time_point> castTime = opponent->buffCastTime(skillId);
      if (castTime) {
        remainingTimeOpponentBuffs_[i] = std::chrono::duration_cast<std::chrono::milliseconds>(entity::Character::BuffData::ClockType::now()-*castTime).count();
      } else {
        LOG(WARNING) << "Buff " << bot.gameData().getSkillName(skillId) << " is active for opponent, but no cast time is known";
        remainingTimeOpponentBuffs_[i] = 0;
      }
    } else {
      remainingTimeOpponentBuffs_[i] = 0;
    }
    // ============================================================================
  }

  static constexpr std::array kDebuffs{packet::enums::AbnormalStateFlag::kShocked, packet::enums::AbnormalStateFlag::kBurnt};
  if (kDebuffs.size() != remainingTimeOurDebuffs_.size() ||
      kDebuffs.size() != remainingTimeOpponentDebuffs_.size()) {
    throw std::runtime_error("Local debuff array and observation's debuff array do not have the same size");
  }
  for (int i=0; i<kDebuffs.size(); ++i) {
    const packet::enums::AbnormalStateFlag debuff = kDebuffs[i];
    // TODO:
    // std::array<int, 2> remainingTimeOurDebuffs_;
    // std::array<int, 2> remainingTimeOpponentDebuffs_;
  }

  static constexpr std::array kSkills{28, 131, 554, 1253, 1256, 1271, 1272, 1281, 1335, 1377, 1380, 1398, 1399, 1410, 1421, 1441, 8312, 21209, 30577, 37, 114, 298, 300, 322, 339, 371, 588, 610, 644, 1315, 1343, 1449};
  if (kSkills.size() != skillCooldowns_.size()) {
    throw std::runtime_error("Local skill array and observation's skill array do not have the same size");
  }
  for (int i=0; i<kSkills.size(); ++i) {
    const sro::scalar_types::ReferenceSkillId skillId = kSkills[i];
    std::optional<std::chrono::milliseconds> remainingCooldown = bot.selfState()->skillEngine.skillRemainingCooldown(skillId, bot.eventBroker());
    skillCooldowns_[i] = remainingCooldown.value_or(std::chrono::milliseconds(0)).count();
  }

  static constexpr std::array kItems{5, 12, 56};
  if (kItems.size() != itemCooldowns_.size()) {
    throw std::runtime_error("Local item array and observation's item array do not have the same size");
  }
  const std::map<type_id::TypeId, broker::EventBroker::EventId> &itemCooldownEventIds = bot.selfState()->getItemCooldownEventIdMap();
  for (int i=0; i<kItems.size(); ++i) {
    const sro::scalar_types::ReferenceObjectId itemId = kItems[i];
    const auto it = itemCooldownEventIds.find(itemId);
    if (it == itemCooldownEventIds.end()) {
      itemCooldowns_[i] = 0;
    } else {
      std::optional<std::chrono::milliseconds> remainingCooldown = bot.eventBroker().timeRemainingOnDelayedEvent(it->second);
      itemCooldowns_[i] = remainingCooldown.value_or(std::chrono::milliseconds(0)).count();
    }
  }
}

std::string Observation::toString() const {
  return absl::StrFormat("{hp:%d/%d, mp:%d/%d, opponentHp:%d/%d, opponentMp:%d/%d, skillCooldowns:[%s], itemCooldowns:[%s]}", ourCurrentHp_,  ourMaxHp_, ourCurrentMp_,  ourMaxMp_, opponentCurrentHp_,  opponentMaxHp_, opponentCurrentMp_,  opponentMaxMp_, absl::StrJoin(skillCooldowns_, ", "), absl::StrJoin(itemCooldowns_, ", "));
}

}