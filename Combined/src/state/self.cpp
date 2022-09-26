#include "helpers.hpp"
#include "logging.hpp"
#include "self.hpp"

// From Pathfinder
#include "math_helpers.h"

#include <silkroad_lib/position_math.h>
#include <silkroad_lib/constants.h>

#include <cmath>
#include <iostream>

namespace state {

Self::Self(broker::EventBroker &eventBroker, const pk2::GameData &gameData) : eventBroker_(eventBroker), gameData_(gameData) {
  auto eventHandleFunction = std::bind(&Self::handleEvent, this, std::placeholders::_1);
  eventBroker_.subscribeToEvent(event::EventCode::kEnteredNewRegion, eventHandleFunction);
}

void Self::handleEvent(const event::Event *event) {
  try {
    const auto eventCode = event->eventCode;
    switch (eventCode) {
      case event::EventCode::kEnteredNewRegion:
        enteredRegion();
        break;
    }
  } catch (std::exception &ex) {
    LOG() << "Error while handling event!\n  " << ex.what() << std::endl;
  }
}

void Self::initialize(uint32_t globalId, uint32_t refObjId) {
  globalId_ = globalId;
  privateSetRaceAndGender(refObjId);

  maxHp_.reset();
  maxMp_.reset();

  spawned_ = true;
  haveOpenedStorageSinceTeleport = false;
}

void Self::setRaceAndGender(uint32_t refObjId) {
  privateSetRaceAndGender(refObjId);
}

void Self::setCurrentLevel(uint8_t currentLevel) {
  currentLevel_ = currentLevel;
}

void Self::setSkillPoints(uint64_t skillPoints) {
  skillPoints_ = skillPoints;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kCharacterSkillPointsUpdated));
}

void Self::setCurrentExpAndSpExp(uint32_t currentExperience, uint32_t currentSpExperience) {
  currentExperience_ = currentExperience;
  currentSpExperience_ = currentSpExperience;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kCharacterExperienceUpdated));
}

void Self::resetHpPotionEventId() {
  hpPotionEventId_.reset();
}

void Self::resetMpPotionEventId() {
  mpPotionEventId_.reset();
}

void Self::resetVigorPotionEventId() {
  vigorPotionEventId_.reset();
}

void Self::resetUniversalPillEventId() {
  universalPillEventId_.reset();
}

void Self::resetPurificationPillEventId() {
  purificationPillEventId_.reset();
}

void Self::setHpPotionEventId(const broker::TimerManager::TimerId &timerId) {
  hpPotionEventId_ = timerId;
}

void Self::setMpPotionEventId(const broker::TimerManager::TimerId &timerId) {
  mpPotionEventId_ = timerId;
}

void Self::setVigorPotionEventId(const broker::TimerManager::TimerId &timerId) {
  vigorPotionEventId_ = timerId;
}

void Self::setUniversalPillEventId(const broker::TimerManager::TimerId &timerId) {
  universalPillEventId_ = timerId;
}

void Self::setPurificationPillEventId(const broker::TimerManager::TimerId &timerId) {
  purificationPillEventId_ = timerId;
}

void Self::setSpeed(float walkSpeed, float runSpeed) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  if (walkSpeed == walkSpeed_ && runSpeed == runSpeed_) {
    // Didnt actually change
    return;
  }
  // Get interpolated position before change speed, since that calulation depends on our current speed
  const auto interpolatedPosition = interpolateCurrentPosition();
  walkSpeed_ = walkSpeed;
  runSpeed_ = runSpeed;
  if (moving_) {
    // In order to be able to interpolate position in the future, we need to update these values
    lastKnownPosition_ = interpolatedPosition;
    startedMovingTime_ = currentTime;

    if (destinationPosition_) {
      if (!movingEventId_) {
        throw std::runtime_error("We're moving towards some desitnation position, but there's no timer running");
      }
      // Update timer for new speed
      eventBroker_.cancelDelayedEvent(*movingEventId_);
      auto seconds = helpers::secondsToTravel(lastKnownPosition_, *destinationPosition_, currentSpeed());
      movingEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMovementTimerEnded), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
    }

    // Publish a movement began event since this is essentially creating a new movement
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMovementBegan));
  }
}

void Self::setHwanSpeed(float hwanSpeed) {
  hwanSpeed_ = hwanSpeed;
}

void Self::setLifeState(entity::LifeState lifeState) {
  lifeState_ = lifeState;
}

void Self::setMotionState(entity::MotionState motionState) {
  motionState_ = motionState;
  if (motionState_ == entity::MotionState::kRun || motionState_ == entity::MotionState::kWalk) {
    // Save whether we were walking or running last
    lastMotionState_ = motionState_;
  }
}

void Self::setBodyState(packet::enums::BodyState bodyState) {
  bodyState_ = bodyState;
}

void Self::setStationaryAtPosition(const sro::Position &position) {
  cancelMovement();
  lastKnownPosition_ = position;
  destinationPosition_.reset();
  movementAngle_.reset();
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMovementEnded));
}

void Self::syncPosition(const sro::Position &position) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  const auto expectedPosition = this->position();
  lastKnownPosition_ = position;
  // TODO: Need angle?
  if (moving_) {
    startedMovingTime_ = currentTime;
    const auto offByDistance = sro::position_math::calculateDistance2D(expectedPosition, position);
    LOG() << "We are moving, syncing our position. We were off by " << offByDistance << std::endl;
    // Might be worth recalculating travel time and starting a new timer
  }
}

void Self::movementTimerCompleted() {
  if (movementAngle_) {
    // A timer shouldnt exist if we're running towards some angle
    throw std::runtime_error("Self: Movement timer completed, but we were running towards some angle");
  }
  if (!movingEventId_) {
    throw std::runtime_error("Self: Movement timer completed, but had no running timer");
  }
  if (!moving_) {
    throw std::runtime_error("Self: Movement timer completed, but we werent moving");
  }
  if (!destinationPosition_) {
    throw std::runtime_error("Self: Movement timer completed, but we dont know where we were going");
  }
  cancelMovement();
  lastKnownPosition_ = *destinationPosition_;
  destinationPosition_.reset();
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMovementEnded));
}

void Self::setMovingToDestination(const std::optional<sro::Position> &sourcePosition, const sro::Position &destinationPosition) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  if (sourcePosition) {
    lastKnownPosition_ = *sourcePosition;
  } else if (moving_) {
    // We've pivoted while moving, calculate where we are and save that
    lastKnownPosition_ = interpolateCurrentPosition();
  }
  if (lastKnownPosition_ == destinationPosition) {
    // Not going anywhere
    setStationaryAtPosition(lastKnownPosition_);
    return;
  }
  cancelMovement();
  moving_ = true;
  startedMovingTime_ = currentTime;
  destinationPosition_ = destinationPosition;
  movementAngle_.reset();

  checkIfWillLeaveRegionAndSetTimer(lastKnownPosition_);

  // Start timer
  const auto seconds = helpers::secondsToTravel(lastKnownPosition_, *destinationPosition_, currentSpeed());
  movingEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMovementTimerEnded), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMovementBegan));
}

void Self::setMovingTowardAngle(const std::optional<sro::Position> &sourcePosition, const uint16_t angle) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  if (sourcePosition) {
    lastKnownPosition_ = *sourcePosition;
  } else if (moving_) {
    // We've pivoted while moving, calculate where we are and save that
    lastKnownPosition_ = interpolateCurrentPosition();
  }
  cancelMovement();
  moving_ = true;
  startedMovingTime_ = currentTime;
  destinationPosition_.reset();
  movementAngle_ = angle;

  checkIfWillLeaveRegionAndSetTimer(lastKnownPosition_);
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMovementBegan));
}

void Self::checkIfWillLeaveRegionAndSetTimer(const sro::Position &currentPosition) {
  if (destinationPosition_) {
    if (currentPosition.regionId() == destinationPosition_->regionId()) {
      // Not going to leave our region
      return;
    }
    // We're going to change regions
    const auto xSectorDiff = destinationPosition_->xSector() - currentPosition.xSector();
    const auto zSectorDiff = destinationPosition_->zSector() - currentPosition.zSector();
    const auto xDiff = (destinationPosition_->xOffset() - currentPosition.xOffset()) + xSectorDiff * 1920.0;
    const auto zDiff = (destinationPosition_->zOffset() - currentPosition.zOffset()) + zSectorDiff * 1920.0;
    calculateTimeUntilCollisionWithRegionBoundaryAndPublishDelayedEvent(currentPosition, xDiff, zDiff);
  } else {
    if (!movementAngle_) {
      throw std::runtime_error("We know we're moving, but not to a destination nor with an angle");
    }
    const auto angleRadians = pathfinder::math::k2Pi * static_cast<double>(*movementAngle_) / std::numeric_limits<uint16_t>::max();
    const double kLenToExtend = sqrt(2 * 1920.0 * 1920.0) + 1;
    const double xAdd = kLenToExtend * cos(angleRadians);
    const double yAdd = kLenToExtend * sin(angleRadians);
    calculateTimeUntilCollisionWithRegionBoundaryAndPublishDelayedEvent(currentPosition, xAdd, yAdd);
  }
}

void Self::calculateTimeUntilCollisionWithRegionBoundaryAndPublishDelayedEvent(const sro::Position &currentPosition, double dx, double dy) {
  pathfinder::Vector trajectoryPoint0{currentPosition.xOffset(), currentPosition.zOffset()}, trajectoryPoint1{currentPosition.xOffset()+dx, currentPosition.zOffset()+dy};
  std::array<std::pair<pathfinder::Vector, pathfinder::Vector>, 4> regionBoundaries = {
    std::make_pair(pathfinder::Vector(0.0, 0.0), pathfinder::Vector(1920.0, 0.0)),
    std::make_pair(pathfinder::Vector(0.0, 0.0), pathfinder::Vector(0.0, 1920.0)),
    std::make_pair(pathfinder::Vector(1920.0, 1920.0), pathfinder::Vector(1920.0, 0.0)),
    std::make_pair(pathfinder::Vector(1920.0, 1920.0), pathfinder::Vector(0.0, 1920.0))
  };
  std::optional<pathfinder::Vector> intersectionPoint = [&]() -> std::optional<pathfinder::Vector> {
    for (const auto &regionBoundary : regionBoundaries) {
      pathfinder::Vector intersectionPoint;
      const auto intRes = pathfinder::math::intersect(trajectoryPoint0, trajectoryPoint1, regionBoundary.first, regionBoundary.second, &intersectionPoint);
      if (intRes == pathfinder::math::IntersectionResult::kOne) {
        return intersectionPoint;
      }
    }
    return {};
  }();
  if (!intersectionPoint) {
    throw std::runtime_error("This must intersect somewhere since we know we're crossing a region boundary");
  }
  sro::Position intersectionPos(currentPosition.regionId(), intersectionPoint->x(), 0.0f, intersectionPoint->y());
  // Start timer
  const auto seconds = helpers::secondsToTravel(currentPosition, intersectionPos, currentSpeed());
  enteredNewRegionEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kEnteredNewRegion), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
}

void Self::enteredRegion() {
  if (!moving_) {
    // No work to do
    LOG() << "[No retrigger] Not moving" << std::endl;
    return;
  }
  // Do this check based on where we currently are
  const auto currentPosition = position();
  checkIfWillLeaveRegionAndSetTimer(currentPosition);
}

void Self::cancelMovement() {
  if (movingEventId_) {
    eventBroker_.cancelDelayedEvent(*movingEventId_);
    movingEventId_.reset();
  }
  if (enteredNewRegionEventId_) {
    eventBroker_.cancelDelayedEvent(*enteredNewRegionEventId_);
    enteredNewRegionEventId_.reset();
  }
  moving_ = false;
}

void Self::setHp(uint32_t hp) {
  hp_ = hp;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kHpChanged));
}

void Self::setMp(uint32_t mp) {
  mp_ = mp;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMpChanged));
}

void Self::setMaxHpMp(uint32_t maxHp, uint32_t maxMp) {
  maxHp_ = maxHp;
  maxMp_ = maxMp;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMaxHpMpChanged));
}

void Self::updateStates(uint32_t stateBitmask, const std::vector<uint8_t> &stateLevels) {
  const auto oldStateBitmask = this->stateBitmask();
  uint32_t newlyReceivedStates = (oldStateBitmask ^ stateBitmask) & stateBitmask;
  uint32_t expiredStates = (oldStateBitmask ^ stateBitmask) & oldStateBitmask;
  this->setStateBitmask(stateBitmask);

  int stateLevelIndex=0;
  if (newlyReceivedStates != 0) {
    // We have some new states!
    for (int bitNum=0; bitNum<32; ++bitNum) {
      const auto kBit = static_cast<uint32_t>(1) << bitNum;
      if ((newlyReceivedStates & kBit) != 0) {
        const auto kState = static_cast<packet::enums::AbnormalStateFlag>(kBit);
        if (kState <= packet::enums::AbnormalStateFlag::kZombie) {
          // Legacy state
          // We now are kState
        } else {
          // Modern state
          // We now are under kState
          this->setModernStateLevel(helpers::fromBitNum(bitNum), stateLevels[stateLevelIndex]);
          ++stateLevelIndex;
        }
      }
    }
  }
  if (expiredStates != 0) {
    // We have some expired states
    for (int bitNum=0; bitNum<32; ++bitNum) {
      const auto kBit = static_cast<uint32_t>(1) << bitNum;
      if ((expiredStates & kBit) != 0) {
        const auto kState = static_cast<packet::enums::AbnormalStateFlag>(kBit);
        if (kState <= packet::enums::AbnormalStateFlag::kZombie) {
          // Legacy state
          // We are no longer kState
        } else {
          // Modern state
          // We are no longer under kState
          this->setModernStateLevel(helpers::fromBitNum(bitNum), 0);
        }
      }
    }
  }
}

void Self::setStateBitmask(uint32_t stateBitmask) {
  stateBitmask_ = stateBitmask;
}

void Self::setLegacyStateEffect(packet::enums::AbnormalStateFlag flag, uint16_t effect) {
  const auto index = helpers::toBitNum(flag);
  legacyStateEffects_[index] = effect;
}

void Self::setModernStateLevel(packet::enums::AbnormalStateFlag flag, uint8_t level) {
  const auto index = helpers::toBitNum(flag);
  modernStateLevels_[index] = level;
}

void Self::setMasteriesAndSkills(const std::vector<packet::structures::Mastery> &masteries,
                                 const std::vector<packet::structures::Skill> &skills) {
  masteries_ = masteries;
  skills_ = skills;
}

void Self::setGold(uint64_t goldAmount) {
  gold_ = goldAmount;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kInventoryGoldUpdated));
}

void Self::setStorageGold(uint64_t goldAmount) {
  storageGold_ = goldAmount;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStorageGoldUpdated));
}

void Self::setGuildStorageGold(uint64_t goldAmount) {
  guildStorageGold_ = goldAmount;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kGuildStorageGoldUpdated));
}

// =========================================================================================================
// =================================================Getters=================================================
// =========================================================================================================

bool Self::spawned() const {
  return spawned_;
}

uint32_t Self::globalId() const {
  return globalId_;
}

Race Self::race() const {
  return race_;
}

Gender Self::gender() const {
  return gender_;
}

uint8_t Self::getCurrentLevel() const {
  return currentLevel_;
}

uint64_t Self::getSkillPoints() const {
  return skillPoints_;
}

uint32_t Self::getCurrentExperience() const {
  return currentExperience_;
}

uint32_t Self::getCurrentSpExperience() const {
  return currentSpExperience_;
}

bool Self::haveHpPotionEventId() const {
  return hpPotionEventId_.has_value();
}

bool Self::haveMpPotionEventId() const {
  return mpPotionEventId_.has_value();
}

bool Self::haveVigorPotionEventId() const {
  return vigorPotionEventId_.has_value();
}

bool Self::haveUniversalPillEventId() const {
  return universalPillEventId_.has_value();
}

bool Self::havePurificationPillEventId() const {
  return purificationPillEventId_.has_value();
}

broker::TimerManager::TimerId Self::getHpPotionEventId() const {
  if (!hpPotionEventId_.has_value()) {
    throw std::runtime_error("Self: Asking for hp potion event id, but we dont have one");
  }
  return hpPotionEventId_.value();
}

broker::TimerManager::TimerId Self::getMpPotionEventId() const {
  if (!mpPotionEventId_.has_value()) {
    throw std::runtime_error("Self: Asking for mp potion event id, but we dont have one");
  }
  return mpPotionEventId_.value();
}

broker::TimerManager::TimerId Self::getVigorPotionEventId() const {
  if (!vigorPotionEventId_.has_value()) {
    throw std::runtime_error("Self: Asking for vigor potion event id, but we dont have one");
  }
  return vigorPotionEventId_.value();
}

broker::TimerManager::TimerId Self::getUniversalPillEventId() const {
  if (!universalPillEventId_.has_value()) {
    throw std::runtime_error("Self: Asking for universal pill event id, but we dont have one");
  }
  return universalPillEventId_.value();
}

broker::TimerManager::TimerId Self::getPurificationPillEventId() const {
  if (!purificationPillEventId_.has_value()) {
    throw std::runtime_error("Self: Asking for purification pill event id, but we dont have one");
  }
  return purificationPillEventId_.value();
}

int Self::getHpPotionDelay() const {
  const bool havePanic = (modernStateLevels_[helpers::toBitNum(packet::enums::AbnormalStateFlag::kPanic)] > 0);
  int delay = potionDelayMs_;
  if (havePanic) {
    delay += 4000;
  }
  return delay;
}

int Self::getMpPotionDelay() const {
  const bool haveCombustion = (modernStateLevels_[helpers::toBitNum(packet::enums::AbnormalStateFlag::kCombustion)] > 0);
  int delay = potionDelayMs_;
  if (haveCombustion) {
    delay += 4000;
  }
  return delay;
}

int Self::getVigorPotionDelay() const {
  return potionDelayMs_;
}

int Self::getGrainDelay() const {
  return 4000;
}

int Self::getUniversalPillDelay() const {
  return 1000;
}

int Self::getPurificationPillDelay() const {
  // TODO: This is wrong
  return 20000;
}

float Self::walkSpeed() const {
  return walkSpeed_;
}

float Self::runSpeed() const {
  return runSpeed_;
}

float Self::hwanSpeed() const {
  return hwanSpeed_;
}

float Self::currentSpeed() const {
  if (motionState_ == entity::MotionState::kRun) {
    return runSpeed_;
  } else if (motionState_ == entity::MotionState::kWalk) {
    return walkSpeed_;
  } else if (motionState_ == entity::MotionState::kStand) {
    if (lastMotionState_) {
      if (*lastMotionState_ == entity::MotionState::kRun) {
        return runSpeed_;
      } else if (*lastMotionState_ == entity::MotionState::kWalk) {
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

entity::LifeState Self::lifeState() const {
  return lifeState_;
}

entity::MotionState Self::motionState() const {
  return motionState_;
}

packet::enums::BodyState Self::bodyState() const {
  return bodyState_;
}

sro::Position Self::position() const {
  return interpolateCurrentPosition();
}

bool Self::moving() const {
  return moving_;
}

bool Self::haveDestination() const {
  return destinationPosition_.has_value();
}

sro::Position Self::destination() const {
  if (!destinationPosition_) {
    throw std::runtime_error("Self: Trying to get destination that does not exist");
  }
  return destinationPosition_.value();
}

uint16_t Self::movementAngle() const {
  if (!movementAngle_) {
    throw std::runtime_error("Self: Trying to get movement angle that does not exist");
  }
  return movementAngle_.value();
}

uint32_t Self::hp() const {
  return hp_;
}

uint32_t Self::mp() const {
  return mp_;
}

std::optional<uint32_t> Self::maxHp() const {
  return maxHp_;
}

std::optional<uint32_t> Self::maxMp() const {
  return maxMp_;
}

uint32_t Self::stateBitmask() const {
  return stateBitmask_;
}

std::array<uint16_t,6> Self::legacyStateEffects() const {
  return legacyStateEffects_;
}

std::array<uint8_t,32> Self::modernStateLevels() const {
  return modernStateLevels_;
}

storage::Storage& Self::getCosInventory(uint32_t globalId) {
  auto cosPairIt = cosInventoryMap.find(globalId);
  if (cosPairIt == cosInventoryMap.end()) {
    throw std::runtime_error("Asking for COS inventory which we don't have");
  }
  return cosPairIt->second;
}

uint64_t Self::getGold() const {
  return gold_;
}
uint64_t Self::getStorageGold() const {
  return storageGold_;
}
uint64_t Self::getGuildStorageGold() const {
  return guildStorageGold_;
}

std::vector<packet::structures::Mastery> Self::masteries() const {
  return masteries_;
}

std::vector<packet::structures::Skill> Self::skills() const {
  return skills_;
}

// =====================================Packets-in-flight state=====================================
// Setters
void Self::popItemFromUsedItemQueueIfNotEmpty() {
  if (!usedItemQueue_.empty()) {
    usedItemQueue_.pop_front();
  }
}

void Self::clearUsedItemQueue() {
  usedItemQueue_.clear();
}

void Self::pushItemToUsedItemQueue(uint8_t inventorySlotNum, uint16_t itemTypeId) {
  usedItemQueue_.emplace_back(inventorySlotNum, itemTypeId);
}

void Self::setUserPurchaseRequest(const packet::structures::ItemMovement &itemMovement) {
  userPurchaseRequest_ = itemMovement;
}

void Self::resetUserPurchaseRequest() {
  userPurchaseRequest_.reset();
}

// Getters
bool Self::usedItemQueueIsEmpty() const {
  return usedItemQueue_.empty();
}

bool Self::itemIsInUsedItemQueue(uint16_t itemTypeId) const {
  for (const auto &usedItem : usedItemQueue_) {
    if (usedItem.itemTypeId == itemTypeId) {
      return true;
    }
  }
  return false;
}

Self::UsedItem Self::getUsedItemQueueFront() const {
  if (usedItemQueue_.empty()) {
    throw std::runtime_error("Self: Trying to get front of used item queue that is empty");
  }
  return usedItemQueue_.front();
}

bool Self::haveUserPurchaseRequest() const {
  return userPurchaseRequest_.has_value();
}

packet::structures::ItemMovement Self::getUserPurchaseRequest() const {
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

sro::Position Self::interpolateCurrentPosition() const {
  if (!moving_) {
    return lastKnownPosition_;
  }
  const auto currentTime = std::chrono::high_resolution_clock::now();
  auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime-startedMovingTime_).count();
  if (destinationPosition_) {
    auto totalDistance = sro::position_math::calculateDistance2D(lastKnownPosition_, *destinationPosition_);
    if (totalDistance < 0.0001 /* TODO: Use a double equal function */) {
      // We're at our destination
      return lastKnownPosition_;
    }
    auto expectedTravelTimeSeconds = totalDistance / currentSpeed();
    double percentTraveled = (elapsedTimeMs/1000.0) / expectedTravelTimeSeconds;
    if (percentTraveled < 0) {
      throw std::runtime_error("Self: Traveled negative distance");
    } else if (percentTraveled == 0) {
      return lastKnownPosition_;
    } else if (percentTraveled == 1) {
      return *destinationPosition_;
    } else {
      const auto resultPos = sro::position_math::interpolateBetweenPoints(lastKnownPosition_, *destinationPosition_, percentTraveled);
      if (percentTraveled > 1) {
        LOG() << "Weird, we're moving, but we've traveled \"past\" our destination (" << percentTraveled*100 << "%)" << std::endl;
        if (percentTraveled == std::numeric_limits<double>::infinity()) {
          throw std::runtime_error("Nooo");
        }
        LOG() << "   Destination: " << destinationPosition_->xOffset() << ',' << destinationPosition_->zOffset() << std::endl;
        LOG() << "   Returning pos " << resultPos.xOffset() << ',' << resultPos.zOffset() << std::endl;
      }
      return resultPos;
    }
  } else if (movementAngle_) {
    float angle = *movementAngle_/static_cast<float>(std::numeric_limits<std::remove_reference_t<decltype(*movementAngle_)>>::max()) * sro::constants::k2Pi;
    float xOffset = std::cos(angle) * currentSpeed() * (elapsedTimeMs/1000.0);
    float zOffset = std::sin(angle) * currentSpeed() * (elapsedTimeMs/1000.0);
    return sro::position_math::createNewPositionWith2dOffset(lastKnownPosition_, xOffset, zOffset);
  } else {
    throw std::runtime_error("Moving but no destination position or movement angle");
  }
}

} // namespace state