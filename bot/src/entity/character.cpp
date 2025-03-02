#include "character.hpp"

#include <stdexcept>

namespace entity {

void Character::setLifeState(sro::entity::LifeState newLifeState) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  const bool changed = lifeState != newLifeState;
  lifeState = newLifeState;
  if (newLifeState == sro::entity::LifeState::kDead) {
    privateSetStationaryAtPosition(interpolateCurrentPosition(currentTime));
  }
  if (changed) {
    if (eventBroker_) {
      eventBroker_->publishEvent<event::EntityLifeStateChanged>(globalId);
    } else {
      LOG(WARNING) << "Trying to publish EntityLifeStateChanged, but don't have event broker";
    }
  }
}

bool Character::currentHpIsKnown() const {
  return currentHp_.has_value();
}

uint32_t Character::currentHp() const {
  return *currentHp_;
}

void Character::setCurrentHp(uint32_t hp) {
  const bool changed = !currentHp_ || *currentHp_ != hp;
  currentHp_ = hp;
  if (changed) {
    if (eventBroker_) {
      eventBroker_->publishEvent<event::EntityHpChanged>(globalId);
    } else {
      LOG(WARNING) << "Trying to publish EntityHpChanged, but don't have event broker";
    }
  }
}

std::set<sro::scalar_types::ReferenceSkillId> Character::activeBuffs() const {
  std::set<sro::scalar_types::ReferenceSkillId> result;
  for (const auto &buffTokenDataPair : buffDataMap) {
    result.emplace(buffTokenDataPair.second.skillRefId);
  }
  return result;
}

bool Character::buffIsActive(sro::scalar_types::ReferenceSkillId skillRefId) const {
  for (const auto &buffTokenDataPair : buffDataMap) {
    if (buffTokenDataPair.second.skillRefId == skillRefId) {
      return true;
    }
  }
  return false;
}

std::optional<Character::BuffData::ClockType::time_point> Character::buffCastTime(sro::scalar_types::ReferenceSkillId skillRefId) const {
  for (const auto &buffTokenDataPair : buffDataMap) {
    if (buffTokenDataPair.second.skillRefId == skillRefId) {
      return buffTokenDataPair.second.castTime;
    }
  }
  throw std::runtime_error(absl::StrFormat("Character::buffCastTime: No buff with skill ID %d", skillRefId));
}

void Character::addBuff(sro::scalar_types::ReferenceSkillId skillRefId, sro::scalar_types::BuffTokenType tokenId, std::optional<BuffData::ClockType::time_point> castTime) {
  if (auto it = buffDataMap.find(tokenId); it != buffDataMap.end()) {
    // Already tracking this buff, nothing to do.
    // We do not overwrite the time of the existing buff data. The first time the buff data was created was most accurate. The first time we try to add this buff is the soonest the server could tell us that this buff is active. Any later packet doesn't give any new info and will be less accurate.
    return;
  }
  buffDataMap[tokenId] = BuffData{skillRefId, castTime};
  if (eventBroker_) {
    eventBroker_->publishEvent<event::BuffAdded>(globalId, skillRefId);
  } else {
    LOG(WARNING) << "Trying to publish BuffAdded, but don't have event broker";
  }
}

void Character::removeBuff(sro::scalar_types::ReferenceSkillId skillRefId, sro::scalar_types::BuffTokenType tokenId) {
  auto buffIt = buffDataMap.find(tokenId);
  if (buffIt == buffDataMap.end()) {
    throw std::runtime_error("Trying to remove buff "+std::to_string(skillRefId)+" from entity, but cannot find buff");
  }
  buffDataMap.erase(buffIt);
  if (eventBroker_) {
    eventBroker_->publishEvent<event::BuffRemoved>(globalId, skillRefId);
  } else {
    LOG(WARNING) << "Trying to publish BuffRemoved, but don't have event broker";
  }
}

void Character::clearBuffs() {
  buffDataMap.clear();
}

} // namespace entity