#include "bot.hpp"
#include "event/event.hpp"
#include "rl/observation.hpp"

#include <absl/algorithm/container.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <array>
#include <ostream>
#include <istream>

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
  timestamp_ = std::chrono::steady_clock::now();
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
        // Get total duration so that we can normalize the remaining time to [0,1]
        const sro::pk2::ref::Skill &skill = bot.gameData().skillData().getSkillById(skillId);
        if (skill.hasParam("DURA")) {
          remainingTimeOurBuffs_[i] = std::clamp(std::chrono::duration_cast<std::chrono::milliseconds>(entity::Character::BuffData::ClockType::now()-*castTime).count() / static_cast<float>(skill.durationMs()), 0.0f, 1.0f);
        } else {
          // So far, I have only seen this for the fire wall skill.
          remainingTimeOurBuffs_[i] = 1.0;
        }
      } else {
        LOG(WARNING) << "Buff " << bot.gameData().getSkillName(skillId) << " is active for us, but no cast time is known";
        remainingTimeOurBuffs_[i] = 0.0;
      }
    } else {
      remainingTimeOurBuffs_[i] = 0.0;
    }
    // ================================= Opponent =================================
    if (opponent->buffIsActive(skillId)) {
      std::optional<entity::Character::BuffData::ClockType::time_point> castTime = opponent->buffCastTime(skillId);
      if (castTime) {
        // Get total duration so that we can normalize the remaining time to [0,1]
        const sro::pk2::ref::Skill &skill = bot.gameData().skillData().getSkillById(skillId);
        if (skill.hasParam("DURA")) {
          remainingTimeOpponentBuffs_[i] = std::clamp(std::chrono::duration_cast<std::chrono::milliseconds>(entity::Character::BuffData::ClockType::now()-*castTime).count() / static_cast<float>(skill.durationMs()), 0.0f, 1.0f);
        } else {
          // So far, I have only seen this for the fire wall skill.
          remainingTimeOpponentBuffs_[i] = 1.0;
        }
      } else {
        LOG(WARNING) << "Buff " << bot.gameData().getSkillName(skillId) << " is active for opponent, but no cast time is known";
        remainingTimeOpponentBuffs_[i] = 0.0;
      }
    } else {
      remainingTimeOpponentBuffs_[i] = 0.0;
    }
    // ============================================================================
  }

  static constexpr std::array kDebuffs{packet::enums::AbnormalStateFlag::kShocked, packet::enums::AbnormalStateFlag::kBurnt};
  static constexpr std::array kDebuffLevels{36, 41, 48,  54, 62, 82};
  if (kDebuffs.size()*kDebuffLevels.size() != remainingTimeOurDebuffs_.size() ||
      kDebuffs.size()*kDebuffLevels.size() != remainingTimeOpponentDebuffs_.size()) {
    throw std::runtime_error("Local debuff array and observation's debuff array do not have the same size");
  }

  const auto &ourLegacyStateEffects = bot.selfState()->legacyStateEffects();
  const auto &ourLegacyStateEndTimes = bot.selfState()->legacyStateEndTimes();
  const auto &ourLegacyStateTotalDurations = bot.selfState()->legacyStateTotalDurations();
  const auto &opponentLegacyStateEffects = opponent->legacyStateEffects();
  const auto &opponentLegacyStateEndTimes = opponent->legacyStateEndTimes();
  const auto &opponentLegacyStateTotalDurations = opponent->legacyStateTotalDurations();
  int index=0;
  for (packet::enums::AbnormalStateFlag debuff : kDebuffs) {
    for (int level : kDebuffLevels) {
      // ==================================== Us ====================================
      if (ourLegacyStateEffects[helpers::toBitNum(debuff)] == level) {
        std::chrono::milliseconds remainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(ourLegacyStateEndTimes[helpers::toBitNum(debuff)] - timestamp_);
        remainingTimeOurDebuffs_[index] = std::clamp(remainingTime.count() / static_cast<float>(ourLegacyStateTotalDurations[helpers::toBitNum(debuff)].count()), 0.0f, 1.0f);
      } else {
        remainingTimeOurDebuffs_[index] = 0.0;
      }
      // ================================= Opponent =================================
      if (opponentLegacyStateEffects[helpers::toBitNum(debuff)] == level) {
        std::chrono::milliseconds remainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(opponentLegacyStateEndTimes[helpers::toBitNum(debuff)] - timestamp_);
        remainingTimeOpponentDebuffs_[index] = std::clamp(remainingTime.count() / static_cast<float>(opponentLegacyStateTotalDurations[helpers::toBitNum(debuff)].count()), 0.0f, 1.0f);
      } else {
        remainingTimeOpponentDebuffs_[index] = 0.0;
      }
      // ============================================================================
      ++index;
    }
  }

  static constexpr std::array kSkills{28, 131, 554, 1253, 1256, 1271, 1272, 1281, 1335, 1377, 1380, 1398, 1399, 1410, 1421, 1441, 8312, 21209, 30577, 37, 114, 298, 300, 322, 339, 371, 588, 610, 644, 1315, 1343, 1449};
  if (kSkills.size() != skillCooldowns_.size()) {
    throw std::runtime_error("Local skill array and observation's skill array do not have the same size");
  }
  for (int i=0; i<kSkills.size(); ++i) {
    const sro::scalar_types::ReferenceSkillId skillId = kSkills[i];
    std::optional<std::chrono::milliseconds> remainingCooldown = bot.selfState()->skillRemainingCooldown(skillId);
    // Get total duration so that we can normalize the remaining time to [0,1]
    const int32_t totalCooldownDurationMs = bot.gameData().skillData().getSkillById(skillId).actionReuseDelay;
    skillCooldowns_[i] = std::clamp(remainingCooldown.value_or(std::chrono::milliseconds(0)).count() / static_cast<float>(totalCooldownDurationMs), 0.0f, 1.0f);
  }

  static constexpr std::array kItems{
    5, // HP Potion
    12, // MP Potion
    56 // Universal Pill
  };
  if (kItems.size() != itemCooldowns_.size()) {
    throw std::runtime_error("Local item array and observation's item array do not have the same size");
  }
  const std::map<type_id::TypeId, broker::EventBroker::EventId> &itemCooldownEventIds = bot.selfState()->getItemCooldownEventIdMap();
  for (int i=0; i<kItems.size(); ++i) {
    const sro::scalar_types::ReferenceObjectId itemId = kItems[i];
    const int32_t totalCooldownDurationMs = [&](){
      if (itemId == 5) {
        return bot.selfState()->getHpPotionDelay();
      } else if (itemId == 12) {
        return bot.selfState()->getMpPotionDelay();
      } else if (itemId == 56) {
        return bot.selfState()->getUniversalPillDelay();
      } else {
        throw std::runtime_error("Unknown item id: " + std::to_string(itemId));
      }
    }();
    const auto it = itemCooldownEventIds.find(itemId);
    if (it == itemCooldownEventIds.end()) {
      itemCooldowns_[i] = 0.0;
    } else {
      std::optional<std::chrono::milliseconds> remainingCooldown = bot.eventBroker().timeRemainingOnDelayedEvent(it->second);
      itemCooldowns_[i] = std::clamp(remainingCooldown.value_or(std::chrono::milliseconds(0)).count() / static_cast<float>(totalCooldownDurationMs), 0.0f, 1.0f);
    }
  }
}

std::string Observation::toString() const {
  return absl::StrFormat("{hp:%d/%d, mp:%d/%d, opponentHp:%d/%d, opponentMp:%d/%d, skillCooldowns:[%s], itemCooldowns:[%s]}", ourCurrentHp_,  ourMaxHp_, ourCurrentMp_,  ourMaxMp_, opponentCurrentHp_,  opponentMaxHp_, opponentCurrentMp_,  opponentMaxMp_, absl::StrJoin(skillCooldowns_, ", "), absl::StrJoin(itemCooldowns_, ", "));
}

template<typename T>
static void writeBinary(std::ostream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template<typename T>
static void readBinary(std::istream &in, T &value) {
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
}

void Observation::saveToStream(std::ostream &out) const {
  using namespace std::chrono;
  const int64_t ts = duration_cast<nanoseconds>(timestamp_.time_since_epoch()).count();
  writeBinary(out, ts);
  writeBinary(out, static_cast<int>(eventCode_));
  writeBinary(out, ourCurrentHp_);
  writeBinary(out, ourMaxHp_);
  writeBinary(out, ourCurrentMp_);
  writeBinary(out, ourMaxMp_);
  writeBinary(out, weAreKnockedDown_);
  writeBinary(out, opponentCurrentHp_);
  writeBinary(out, opponentMaxHp_);
  writeBinary(out, opponentCurrentMp_);
  writeBinary(out, opponentMaxMp_);
  writeBinary(out, opponentIsKnockedDown_);
  writeBinary(out, hpPotionCount_);
  for (float v : remainingTimeOurBuffs_) writeBinary(out, v);
  for (float v : remainingTimeOpponentBuffs_) writeBinary(out, v);
  for (float v : remainingTimeOurDebuffs_) writeBinary(out, v);
  for (float v : remainingTimeOpponentDebuffs_) writeBinary(out, v);
  for (float v : skillCooldowns_) writeBinary(out, v);
  for (float v : itemCooldowns_) writeBinary(out, v);
}

Observation Observation::loadFromStream(std::istream &in) {
  Observation obs;
  using namespace std::chrono;
  int64_t ts;
  readBinary(in, ts);
  obs.timestamp_ = steady_clock::time_point(nanoseconds(ts));
  int ev;
  readBinary(in, ev);
  obs.eventCode_ = static_cast<event::EventCode>(ev);
  readBinary(in, obs.ourCurrentHp_);
  readBinary(in, obs.ourMaxHp_);
  readBinary(in, obs.ourCurrentMp_);
  readBinary(in, obs.ourMaxMp_);
  readBinary(in, obs.weAreKnockedDown_);
  readBinary(in, obs.opponentCurrentHp_);
  readBinary(in, obs.opponentMaxHp_);
  readBinary(in, obs.opponentCurrentMp_);
  readBinary(in, obs.opponentMaxMp_);
  readBinary(in, obs.opponentIsKnockedDown_);
  readBinary(in, obs.hpPotionCount_);
  for (float &v : obs.remainingTimeOurBuffs_) readBinary(in, v);
  for (float &v : obs.remainingTimeOpponentBuffs_) readBinary(in, v);
  for (float &v : obs.remainingTimeOurDebuffs_) readBinary(in, v);
  for (float &v : obs.remainingTimeOpponentDebuffs_) readBinary(in, v);
  for (float &v : obs.skillCooldowns_) readBinary(in, v);
  for (float &v : obs.itemCooldowns_) readBinary(in, v);
  return obs;
}

}