#include "helpers.hpp"
#include "logging.hpp"
#include "self.hpp"

#include "type_id/categories.hpp"

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

void Self::initialize(sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceObjectId refObjId, uint32_t jId) {
  this->globalId = globalId;
  this->refObjId = refObjId;
  this->jId = jId;
  setRaceAndGender();

  maxHp_.reset();
  maxMp_.reset();

  spawned_ = true;
  haveOpenedStorageSinceTeleport = false;
}

void Self::initializeCurrentHp(uint32_t hp) {
  currentHp_ = hp;
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

void Self::setHwanSpeed(float hwanSpeed) {
  hwanSpeed_ = hwanSpeed;
}

void Self::setBodyState(packet::enums::BodyState bodyState) {
  bodyState_ = bodyState;
}

void Self::setMovingToDestination(const std::optional<sro::Position> &sourcePosition, const sro::Position &destinationPosition, broker::EventBroker &eventBroker) {
  entity::MobileEntity::setMovingToDestination(sourcePosition, destinationPosition, eventBroker);
  checkIfWillLeaveRegionAndSetTimer(eventBroker);
}

void Self::setMovingTowardAngle(const std::optional<sro::Position> &sourcePosition, const sro::Angle angle, broker::EventBroker &eventBroker) {
  entity::MobileEntity::setMovingTowardAngle(sourcePosition, angle, eventBroker);
  checkIfWillLeaveRegionAndSetTimer(eventBroker);
}

void Self::checkIfWillLeaveRegionAndSetTimer(broker::EventBroker &eventBroker) {
  if (!moving()) {
    // Not moving, nothing to do
    return;
  }
  const auto currentPosition = position();
  if (destinationPosition) {
    if (currentPosition.regionId() == destinationPosition->regionId()) {
      // Not going to leave our region
      return;
    }
    // We're going to change regions
    const auto xSectorDiff = destinationPosition->xSector() - currentPosition.xSector();
    const auto zSectorDiff = destinationPosition->zSector() - currentPosition.zSector();
    const auto xDiff = (destinationPosition->xOffset() - currentPosition.xOffset()) + xSectorDiff * 1920.0;
    const auto zDiff = (destinationPosition->zOffset() - currentPosition.zOffset()) + zSectorDiff * 1920.0;
    calculateTimeUntilCollisionWithRegionBoundaryAndPublishDelayedEvent(currentPosition, xDiff, zDiff, eventBroker);
  } else {
    const auto angleRadians = pathfinder::math::k2Pi * static_cast<double>(angle_) / std::numeric_limits<uint16_t>::max();
    const double kLenToExtend = sqrt(2 * 1920.0 * 1920.0) + 1;
    const double xAdd = kLenToExtend * cos(angleRadians);
    const double yAdd = kLenToExtend * sin(angleRadians);
    calculateTimeUntilCollisionWithRegionBoundaryAndPublishDelayedEvent(currentPosition, xAdd, yAdd, eventBroker);
  }
}

void Self::calculateTimeUntilCollisionWithRegionBoundaryAndPublishDelayedEvent(const sro::Position &currentPosition, double dx, double dy, broker::EventBroker &eventBroker) {
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
  enteredNewRegionEventId_ = eventBroker.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kEnteredNewRegion), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
}

void Self::enteredRegion() {
  if (!moving()) {
    // No work to do
    return;
  }
  // Do this check based on where we currently are
  const auto currentPosition = position();
  checkIfWillLeaveRegionAndSetTimer(eventBroker_);
}

void Self::cancelMovement(broker::EventBroker &eventBroker) {
  entity::MobileEntity::cancelMovement(eventBroker);
  if (enteredNewRegionEventId_) {
    eventBroker_.cancelDelayedEvent(*enteredNewRegionEventId_);
    enteredNewRegionEventId_.reset();
  }
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
        LOG() << "We are now under " << kState << std::endl;
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
        LOG() << "We are no longer under " << kState << std::endl;
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

void Self::usedAnItem(type_id::TypeId typeData, broker::EventBroker &eventBroker) {
  // For now, we figure out the cooldown duration right here. Maybe in the future, it should be passed into this function
  if (itemCooldownEventIdMap_.find(typeData) != itemCooldownEventIdMap_.end()) {
    throw std::runtime_error("Trying to use an item, but it's already on cooldown");
  }

  // TODO: We should move this to a more global configuration area for general bot mechanics configuration
  //       Maybe we could try to improve this value based on item use results
  static const int kPotionDelayBufferMs_ = 0; //200 too fast sometimes, 300 seems always good, had 225

  std::optional<int> cooldownMilliseconds;
  if (type_id::categories::kHpPotion.contains(typeData)) {
    cooldownMilliseconds = getHpPotionDelay() + kPotionDelayBufferMs_;
  } else if (type_id::categories::kMpPotion.contains(typeData)) {
    cooldownMilliseconds = getMpPotionDelay() + kPotionDelayBufferMs_;
  } else if (type_id::categories::kVigorPotion.contains(typeData)) {
    cooldownMilliseconds = getVigorPotionDelay() + kPotionDelayBufferMs_;
  } else if (type_id::categories::kUniversalPill.contains(typeData)) {
    cooldownMilliseconds = getUniversalPillDelay();
  } else if (type_id::categories::kPurificationPill.contains(typeData)) {
    cooldownMilliseconds = getPurificationPillDelay();
  }

  if (!cooldownMilliseconds) {
    LOG() << "Used an item (" << type_id::toString(typeData) << "), but we don't know its cooldown time." << std::endl;
    return;
  }

  // Publish a delayed event
  const auto itemCooldownDelayedEventId = eventBroker.publishDelayedEvent(std::make_unique<event::ItemCooldownEnded>(typeData), std::chrono::milliseconds(*cooldownMilliseconds));
  itemCooldownEventIdMap_.emplace(typeData, itemCooldownDelayedEventId);
}

void Self::itemCooldownEnded(type_id::TypeId itemTypeData) {
  auto it = itemCooldownEventIdMap_.find(itemTypeData);
  if (it != itemCooldownEventIdMap_.end()) {
    itemCooldownEventIdMap_.erase(it);
  } else {
    LOG() << "Item cooldown ended, but we're not tracking it!?" << std::endl;
  }
}

// =========================================================================================================
// =================================================Getters=================================================
// =========================================================================================================

entity::EntityType Self::entityType() const {
  return entity::EntityType::kSelf;
}

bool Self::spawned() const {
  return spawned_;
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
  // TODO: Here is where we should handle if we choose at abuse the pill cooldown bug
  return 50; // TODO: (is this too hacky?) Add a tiny cooldown just so that we don't spam the pill before we even get our next status update
  // TODO: This is wrong (after-the-fact-comment: then why did i use it and where did i get it from? and how did i know it was wrong?)
  //  return 20000;
}

float Self::hwanSpeed() const {
  return hwanSpeed_;
}

packet::enums::BodyState Self::bodyState() const {
  return bodyState_;
}

uint32_t Self::currentMp() const {
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

bool Self::canUseItems() const {
  // TODO
  //  Are we in a state where we cant use items?
  if (!spawned_) {
    // Cannot use items if we're not spawned
    return false;
  }
  return true;
}

bool Self::canUseItem(type_id::TypeCategory itemType) const {
  const bool itemIsOnCooldown = (itemCooldownEventIdMap_.find(itemType.getTypeId()) != itemCooldownEventIdMap_.end());
  if (itemIsOnCooldown) {
    return false;
  }
  // TODO: Other reasons?
  return true;
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

void Self::pushItemToUsedItemQueue(sro::scalar_types::StorageIndexType inventorySlotNum, type_id::TypeId typeId) {
  usedItemQueue_.emplace_back(inventorySlotNum, typeId);
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

bool Self::itemIsInUsedItemQueue(type_id::TypeId typeId) const {
  // TODO: Convert to new TypeId stuff
  for (const auto &usedItem : usedItemQueue_) {
    if (usedItem.typeId == typeId) {
      return true;
    }
  }
  return false;
}

void Self::removedItemFromUsedItemQueue(sro::scalar_types::StorageIndexType inventorySlotNum, type_id::TypeId typeId) {
  const auto it = std::find_if(usedItemQueue_.begin(), usedItemQueue_.end(), [&inventorySlotNum, &typeId](const auto &usedItem) {
    return usedItem.inventorySlotNum == inventorySlotNum && usedItem.typeId == typeId;
  });
  if (it == usedItemQueue_.end()) {
    throw std::runtime_error("Trying to remove item from used item queue which is not present");
  }
  usedItemQueue_.erase(it);
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

void Self::setTrainingAreaGeometry(std::unique_ptr<entity::Circle> &&geometry) {
  trainingAreaGeometry = std::move(geometry);
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kTrainingAreaSet));
}

void Self::resetTrainingAreaGeometry() {
  trainingAreaGeometry.reset();
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kTrainingAreaReset));
}

// =================================================================================================

void Self::addBuff(sro::scalar_types::ReferenceObjectId skillRefId, broker::EventBroker &eventBroker) {
  buffs.emplace(skillRefId);
  LOG() << "Added buff " << skillRefId << " to self" << std::endl;
}

void Self::removeBuff(sro::scalar_types::ReferenceObjectId skillRefId, broker::EventBroker &eventBroker) {
  auto buffIt = buffs.find(skillRefId);
  if (buffIt == buffs.end()) {
    throw std::runtime_error("Tracked buff for ourself, but we dont actually have this buff active");
  }
  buffs.erase(buffIt);
  LOG() << "Removed buff " << skillRefId << " from self" << std::endl;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kOurBuffRemoved));
}

// =================================================================================================

void Self::setRaceAndGender() {
  const auto &character = gameData_.characterData().getCharacterById(refObjId);

  // Set race based on country code 0 is chinese, 1 is european, not sure what 3 is
  if (character.country == 0) {
    race_ = Race::kChinese;
  } else if (character.country == 1) {
    race_ = Race::kEuropean;
  } else {
    throw std::runtime_error("Unknown country character code");
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
  } else {
    throw std::runtime_error("Unknown race");
  }
}

} // namespace state