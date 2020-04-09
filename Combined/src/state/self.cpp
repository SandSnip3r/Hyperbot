#include "self.hpp"

#include <iostream>

namespace state {

int toBitNum(packet::enums::AbnormalStateFlag stateFlag) {
  uint32_t num = static_cast<uint32_t>(stateFlag);
  for (uint32_t i=0; i<32; ++i) {
    if (num & (1<<i)) {
      return i;
    }
  }
  throw std::runtime_error("Tried to get bit for a state "+static_cast<int>(stateFlag));
}

packet::enums::AbnormalStateFlag fromBitNum(int n) {
  return static_cast<packet::enums::AbnormalStateFlag>(uint32_t(1) << n);
}

Self::Self(const pk2::GameData &gameData) : gameData_(gameData) {}

void Self::initialize(uint32_t globalId, uint32_t refObjId, uint32_t hp, uint32_t mp, const std::vector<packet::structures::Mastery> &masteries, const std::vector<packet::structures::Skill> &skills) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  spawned_ = true;
  
  globalId_ = globalId;
  setRaceAndGender(refObjId);

  hp_ = hp;
  mp_ = mp;
  maxHp_.reset();
  maxMp_.reset();

  masteries_ = masteries;
  skills_ = skills;
}

void Self::setLifeState(packet::enums::LifeState lifeState) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  lifeState_ = lifeState;
}

void Self::setBodyState(packet::enums::BodyState bodyState) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  bodyState_ = bodyState;
}

void Self::setHp(uint32_t hp) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  hp_ = hp;
}

void Self::setMp(uint32_t mp) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  mp_ = mp;
}

void Self::setMaxHpMp(uint32_t maxHp, uint32_t maxMp) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  maxHp_ = maxHp;
  maxMp_ = maxMp;
}

void Self::setStateBitmask(uint32_t stateBitmask) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  stateBitmask_ = stateBitmask;
}

void Self::setLegacyStateEffect(packet::enums::AbnormalStateFlag flag, uint16_t effect) {
  const auto index = toBitNum(flag);
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  legacyStateEffects_[index] = effect;
}

void Self::setModernStateLevel(packet::enums::AbnormalStateFlag flag, uint8_t level) {
  const auto index = toBitNum(flag);
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  modernStateLevels_[index] = level;
}

bool Self::spawned() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return spawned_;
}

uint32_t Self::globalId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return globalId_;
}

Race Self::race() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return race_;
}

Gender Self::gender() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return gender_;
}

packet::enums::LifeState Self::lifeState() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return lifeState_;
}

packet::enums::BodyState Self::bodyState() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return bodyState_;
}

uint32_t Self::hp() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return hp_;
}

uint32_t Self::mp() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return mp_;
}

std::optional<uint32_t> Self::maxHp() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return maxHp_;
}

std::optional<uint32_t> Self::maxMp() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return maxMp_;
}

uint32_t Self::stateBitmask() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return stateBitmask_;
}

std::array<uint16_t,6> Self::legacyStateEffects() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return legacyStateEffects_;
}

std::array<uint8_t,32> Self::modernStateLevels() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return modernStateLevels_;
}

std::vector<packet::structures::Mastery> Self::masteries() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return masteries_;
}

std::vector<packet::structures::Skill> Self::skills() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return skills_;
}

void Self::setRaceAndGender(uint32_t refObjId) {
  const auto &gameCharacterData = gameData_.characterData();
  if (!gameCharacterData.haveCharacterWithId(refObjId)) {
    std::cout << "Unable to determine race or gender. No \"item\" data for id: " << refObjId << '\n';
    return;
  }
  const auto &character = gameCharacterData.getCharacterById(refObjId);
  if (character.country == 0) {
    race_ = Race::kChinese;
  } else {
    race_ = Race::kEuropean;
  }
  if (character.charGender == 1) {
    gender_ = Gender::kMale;
  } else {
    gender_ = Gender::kFemale;
  }
}

} // namespace state