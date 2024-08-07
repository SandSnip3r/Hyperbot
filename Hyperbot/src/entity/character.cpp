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
  currentHp_ = hp;
  if (eventBroker_) {
    eventBroker_->publishEvent<event::EntityHpChanged>(globalId);
  } else {
    LOG(WARNING) << "Trying to publish EntityHpChanged, but don't have event broker";
  }
}

std::set<sro::scalar_types::ReferenceObjectId> Character::activeBuffs() const {
  std::set<sro::scalar_types::ReferenceObjectId> result;
  for (const auto &buffTokenDataPair : buffDataMap) {
    result.emplace(buffTokenDataPair.second.skillRefId);
  }
  return result;
}

bool Character::buffIsActive(sro::scalar_types::ReferenceObjectId skillRefId) const {
  for (const auto &buffTokenDataPair : buffDataMap) {
    if (buffTokenDataPair.second.skillRefId == skillRefId) {
      return true;
    }
  }
  return false;
}

int Character::buffMsRemaining(sro::scalar_types::ReferenceObjectId skillRefId) const {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  std::optional<int> maxTime;
  // Loop over all buffs we have and return the remaining time of the buff that will expire the latest.
  for (const auto &buffTokenDataPair : buffDataMap) {
    if (buffTokenDataPair.second.skillRefId == skillRefId) {
      const auto diff = buffTokenDataPair.second.endTimePoint - currentTime;
      const auto diffMs = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
      if (!maxTime || diffMs > *maxTime) {
        maxTime = diffMs;
      }
    }
  }
  if (!maxTime) {
    // Buff is not active.
    throw std::runtime_error("Cannot get remaining time for buff, buff is not active");
  }
  return *maxTime;
}

void Character::addBuff(sro::scalar_types::ReferenceObjectId skillRefId, uint32_t tokenId, int32_t durationMs) {
  if (buffDataMap.find(tokenId) != buffDataMap.end()) {
    throw std::runtime_error("Already tracking buff with this token ID");
  }
  const auto endTimePoint = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(durationMs);
  buffDataMap[tokenId] = BuffData{skillRefId, endTimePoint};
  if (eventBroker_) {
    eventBroker_->publishEvent<event::BuffAdded>(globalId, skillRefId);
  } else {
    LOG(WARNING) << "Trying to publish BuffAdded, but don't have event broker";
  }
}

void Character::removeBuff(sro::scalar_types::ReferenceObjectId skillRefId, uint32_t tokenId) {
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