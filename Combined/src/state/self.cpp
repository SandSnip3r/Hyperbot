#include "self.hpp"
#include "../math/position.hpp"

#include <cmath>
#include <iostream>

namespace state {

// TODO: Replace with constexpr version
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

void Self::initialize(uint32_t globalId,
                      uint32_t refObjId,
                      uint32_t hp,
                      uint32_t mp,
                      const std::vector<packet::structures::Mastery> &masteries,
                      const std::vector<packet::structures::Skill> &skills) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  spawned_ = true;
  
  globalId_ = globalId;
  privateSetRaceAndGender(refObjId);

  hp_ = hp;
  mp_ = mp;
  maxHp_.reset();
  maxMp_.reset();

  masteries_ = masteries;
  skills_ = skills;
}

void Self::setRaceAndGender(uint32_t refObjId) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  privateSetRaceAndGender(refObjId);
}

void Self::resetHpPotionEventId() {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  hpPotionEventId_.reset();
}

void Self::resetMpPotionEventId() {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  mpPotionEventId_.reset();
}

void Self::resetVigorPotionEventId() {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  vigorPotionEventId_.reset();
}

void Self::resetUniversalPillEventId() {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  universalPillEventId_.reset();
}

void Self::resetPurificationPillEventId() {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  purificationPillEventId_.reset();
}

void Self::setHpPotionEventId(const broker::TimerManager::TimerId &timerId) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  hpPotionEventId_ = timerId;
}

void Self::setMpPotionEventId(const broker::TimerManager::TimerId &timerId) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  mpPotionEventId_ = timerId;
}

void Self::setVigorPotionEventId(const broker::TimerManager::TimerId &timerId) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  vigorPotionEventId_ = timerId;
}

void Self::setUniversalPillEventId(const broker::TimerManager::TimerId &timerId) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  universalPillEventId_ = timerId;
}

void Self::setPurificationPillEventId(const broker::TimerManager::TimerId &timerId) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  purificationPillEventId_ = timerId;
}

void Self::setSpeed(float walkSpeed, float runSpeed) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (walkSpeed == walkSpeed_ && runSpeed == runSpeed_) {
    // Didnt actually change
    return;
  }
  if (moving_) {
    // In order to be able to interpolate position in the future, we need to update these values
    lastKnownPosition_ = interpolateCurrentPosition();
    startedMovingTime_ = currentTime;
  }
  walkSpeed_ = walkSpeed;
  runSpeed_ = runSpeed;
}

void Self::setHwanSpeed(float hwanSpeed) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  hwanSpeed_ = hwanSpeed;
}

void Self::setLifeState(packet::enums::LifeState lifeState) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  lifeState_ = lifeState;
}

void Self::setMotionState(packet::enums::MotionState motionState) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  motionState_ = motionState;
  if (motionState_ == packet::enums::MotionState::kRun || motionState_ == packet::enums::MotionState::kWalk) {
    // Save whether we were walking or running last
    lastMotionState_ = motionState_;
  }
}

void Self::setBodyState(packet::enums::BodyState bodyState) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  bodyState_ = bodyState;
}

void Self::setPosition(const packet::structures::Position &position) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  lastKnownPosition_ = position;
  moving_ = false;
  destinationPosition_.reset();
  movementAngle_.reset();
}

void Self::syncPosition(const packet::structures::Position &position) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  lastKnownPosition_ = position;
  // TODO: Need angle?
  if (moving_) {
    startedMovingTime_ = currentTime;
  }
}

void Self::doneMoving() {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (!moving_) {
    throw std::runtime_error("Self: Done moving, but we werent moving");
  }
  if (!destinationPosition_ && !movementAngle_) {
    throw std::runtime_error("Self: Done moving, but we dont know where we were going");
  }
  moving_ = false;
  lastKnownPosition_ = *destinationPosition_;
  destinationPosition_.reset();
  movementAngle_.reset();
}

void Self::setMoving(const packet::structures::Position &destination) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (moving_) {
    // We've pivoted while moving, calculate where we are and save that
    lastKnownPosition_ = interpolateCurrentPosition();
  }
  moving_ = true;
  startedMovingTime_ = currentTime;
  destinationPosition_ = destination;
  movementAngle_.reset();
}

void Self::setMoving(const uint16_t angle) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (moving_) {
    // We've pivoted while moving, calculate where we are and save that
    lastKnownPosition_ = interpolateCurrentPosition();
  }
  moving_ = true;
  startedMovingTime_ = currentTime;
  movementAngle_ = angle;
  destinationPosition_.reset();
}

void Self::setMovingEventId(const broker::TimerManager::TimerId &timerId) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  movingEventId_ = timerId;
}

void Self::resetMovingEventId() {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  movingEventId_.reset();
}

bool Self::haveMovingEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return movingEventId_.has_value();
}

broker::TimerManager::TimerId Self::getMovingEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (!movingEventId_.has_value()) {
    throw std::runtime_error("Self: Asking for moving event id, but we dont have one");
  }
  return movingEventId_.value();
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

void Self::setGold(uint64_t gold) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  gold_ = gold;
}

void Self::addGold(uint64_t gold) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  gold_ += gold;
}

void Self::subtractGold(uint64_t gold) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  gold_ -= gold;
}

void Self::setStorageGold(uint64_t gold) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  storageGold_ = gold;
}

void Self::setGuildStorageGold(uint64_t gold) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  guildStorageGold_ = gold;
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

bool Self::haveHpPotionEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return hpPotionEventId_.has_value();
}

bool Self::haveMpPotionEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return mpPotionEventId_.has_value();
}

bool Self::haveVigorPotionEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return vigorPotionEventId_.has_value();
}

bool Self::haveUniversalPillEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return universalPillEventId_.has_value();
}

bool Self::havePurificationPillEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return purificationPillEventId_.has_value();
}

broker::TimerManager::TimerId Self::getHpPotionEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (!hpPotionEventId_.has_value()) {
    throw std::runtime_error("Self: Asking for hp potion event id, but we dont have one");
  }
  return hpPotionEventId_.value();
}

broker::TimerManager::TimerId Self::getMpPotionEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (!mpPotionEventId_.has_value()) {
    throw std::runtime_error("Self: Asking for mp potion event id, but we dont have one");
  }
  return mpPotionEventId_.value();
}

broker::TimerManager::TimerId Self::getVigorPotionEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (!vigorPotionEventId_.has_value()) {
    throw std::runtime_error("Self: Asking for vigor potion event id, but we dont have one");
  }
  return vigorPotionEventId_.value();
}

broker::TimerManager::TimerId Self::getUniversalPillEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (!universalPillEventId_.has_value()) {
    throw std::runtime_error("Self: Asking for universal pill event id, but we dont have one");
  }
  return universalPillEventId_.value();
}

broker::TimerManager::TimerId Self::getPurificationPillEventId() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (!purificationPillEventId_.has_value()) {
    throw std::runtime_error("Self: Asking for purification pill event id, but we dont have one");
  }
  return purificationPillEventId_.value();
}

int Self::getHpPotionDelay() const {
  const bool havePanic = (modernStateLevels_[toBitNum(packet::enums::AbnormalStateFlag::kPanic)] > 0);
  int delay = potionDelayMs_;
  if (havePanic) {
    delay += 4000;
  }
  return delay;
}

int Self::getMpPotionDelay() const {
  const bool haveCombustion = (modernStateLevels_[toBitNum(packet::enums::AbnormalStateFlag::kCombustion)] > 0);
  int delay = potionDelayMs_;
  if (haveCombustion) {
    delay += 4000;
  }
  return delay;
}

int Self::getVigorPotionDelay() const {
  return potionDelayMs_;
}

float Self::walkSpeed() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return walkSpeed_;
}

float Self::runSpeed() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return runSpeed_;
}

float Self::hwanSpeed() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return hwanSpeed_;
}

float Self::currentSpeed() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return internal_speed();
}

packet::enums::LifeState Self::lifeState() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return lifeState_;
}

packet::enums::MotionState Self::motionState() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return motionState_;
}

packet::enums::BodyState Self::bodyState() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return bodyState_;
}

packet::structures::Position Self::position() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return interpolateCurrentPosition();
}

bool Self::moving() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return moving_;
}

bool Self::haveDestination() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return destinationPosition_.has_value();
}

packet::structures::Position Self::destination() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (!destinationPosition_) {
    throw std::runtime_error("Self: Trying to get destination that does not exist");
  }
  return destinationPosition_.value();
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

storage::Storage& Self::getInventory() {
  // The inventory object holds a reference to our mutex, so it is threadsafe
  return inventory_;
}

storage::BuybackQueue& Self::getBuybackQueue() {
  // The BuybackQueue object holds a reference to our mutex, so it is threadsafe
  return buybackQueue_;
}

uint64_t Self::getGold() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return gold_;
}

uint64_t Self::getStorageGold() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return storageGold_;
}

uint64_t Self::getGuildStorageGold() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return guildStorageGold_;
}

// =====================================Packets-in-flight state=====================================
// Setters
void Self::popItemFromUsedItemQueueIfNotEmpty() {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (!usedItemQueue_.empty()) {
    usedItemQueue_.pop_front();
  }
}

void Self::clearUsedItemQueue() {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  usedItemQueue_.clear();
}

void Self::pushItemToUsedItemQueue(uint8_t inventorySlotNum, uint16_t itemTypeId) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  usedItemQueue_.emplace_back(inventorySlotNum, itemTypeId);
}

void Self::setUserPurchaseRequest(const packet::parsing::ItemMovement &itemMovement) {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  userPurchaseRequest_ = itemMovement;
}

void Self::resetUserPurchaseRequest() {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  userPurchaseRequest_.reset();
}

// Getters
bool Self::usedItemQueueIsEmpty() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return usedItemQueue_.empty();
}

bool Self::itemIsInUsedItemQueue(uint16_t itemTypeId) const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  for (const auto &usedItem : usedItemQueue_) {
    if (usedItem.itemTypeId == itemTypeId) {
      return true;
    }
  }
  return false;
}

Self::UsedItem Self::getUsedItemQueueFront() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (usedItemQueue_.empty()) {
    throw std::runtime_error("Self: Trying to get front of used item queue that is empty");
  }
  return usedItemQueue_.front();
}

bool Self::haveUserPurchaseRequest() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  return userPurchaseRequest_.has_value();
}

packet::parsing::ItemMovement Self::getUserPurchaseRequest() const {
  std::unique_lock<std::mutex> selfLock(selfMutex_);
  if (!userPurchaseRequest_.has_value()) {
    throw std::runtime_error("Self: Trying to get user purchase request that does not exist");
  }
  return userPurchaseRequest_.value();
}

// =================================================================================================

void Self::privateSetRaceAndGender(uint32_t refObjId) {
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

  if (race_ == Race::kChinese) {
    potionDelayMs_ = kChPotionDefaultDelayMs_;
  } else if (race_ == Race::kEuropean) {
    potionDelayMs_ = kEuPotionDefaultDelayMs_;
  }
}

packet::structures::Position Self::interpolateCurrentPosition() const {
  if (!moving_) {
    return lastKnownPosition_;
  }
  const auto currentTime = std::chrono::high_resolution_clock::now();
  auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime-startedMovingTime_).count();
  if (destinationPosition_) {
    auto totalDistance = math::position::calculateDistance(lastKnownPosition_, *destinationPosition_);
    if (totalDistance < 0.0001 /* TODO: Use a double equal function */) {
      // We're at our destination
      return lastKnownPosition_;
    }
    auto expectedTravelTimeSeconds = totalDistance / internal_speed();
    double percentTraveled = (elapsedTimeMs/1000.0) / expectedTravelTimeSeconds;
    if (percentTraveled < 0) {
      throw std::runtime_error("Self: Traveled negative distance");
    } else if (percentTraveled == 0) {
      return lastKnownPosition_;
    } else if (percentTraveled == 1) {
      return *destinationPosition_;
    } else {
      const auto resultPos = math::position::interpolateBetweenPoints(lastKnownPosition_, *destinationPosition_, percentTraveled);
      if (percentTraveled > 1) {
        std::cout << "Weird, we're moving, but we've traveled \"past\" our destination (" << percentTraveled*100 << "%)\n";
        if (percentTraveled == std::numeric_limits<double>::infinity()) {
          throw std::runtime_error("Nooo");
        }
        std::cout << "   Destination: " << destinationPosition_->xOffset << ',' << destinationPosition_->zOffset << "\n";
        std::cout << "   Returning pos " << resultPos.xOffset << ',' << resultPos.zOffset << '\n';
      }
      return resultPos;
    }
  } else if (movementAngle_) {
    float angle = *movementAngle_/static_cast<float>(std::numeric_limits<std::remove_reference_t<decltype(*movementAngle_)>>::max()) * 2*math::kPi;
    float xOffset = std::cos(angle) * internal_speed() * (elapsedTimeMs/1000.0);
    float zOffset = std::sin(angle) * internal_speed() * (elapsedTimeMs/1000.0);
    return math::position::offset(lastKnownPosition_, xOffset, zOffset);
  } else {
    throw std::runtime_error("Moving but no destination position or movement angle");
  }
}

float Self::internal_speed() const {
  if (motionState_ == packet::enums::MotionState::kRun) {
    return runSpeed_;
  } else if (motionState_ == packet::enums::MotionState::kWalk) {
    return walkSpeed_;
  } else if (motionState_ == packet::enums::MotionState::kStand) {
    if (lastMotionState_) {
      if (*lastMotionState_ == packet::enums::MotionState::kRun) {
        return runSpeed_;
      } else if (*lastMotionState_ == packet::enums::MotionState::kWalk) {
        return walkSpeed_;
      } else {
        throw std::runtime_error("Motion state is Stand, last motion state isnt walk or run ("+std::to_string(static_cast<int>(*lastMotionState_))+")");
      }
    } else {
      // Motion  State: Stand, no previous motion state assuming that we're running
      return runSpeed_;
    }
    return runSpeed_;
  } else {
    // TODO: Understand what the other cases are here
    // TODO: Include zerk
    throw std::runtime_error("Trying to get speed, but not walking nor running");
  }
}

} // namespace state