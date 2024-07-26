#include "helpers.hpp"
#include "self.hpp"

#include "type_id/categories.hpp"

// From Pathfinder
#include "math_helpers.h"

#include <silkroad_lib/position_math.h>
#include <silkroad_lib/constants.h>

#include <absl/log/log.h>

#include <cmath>

namespace state {

namespace {

template<typename T>
bool optionalValueIsDifferentFromNewValue(const std::optional<T> &oldOptional, const T &newValue) {
  if (!oldOptional.has_value()) {
    return true;
  }
  return *oldOptional != newValue;
}

} // anonymous namespace

Self::Self(broker::EventBroker &eventBroker, const pk2::GameData &gameData) : eventBroker_(eventBroker), gameData_(gameData) {
  // TODO: Move this out of here. Move it into the Bot
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
    LOG(INFO) << "Error while handling event!\n  " << ex.what();
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
  eventBroker_.publishEvent(event::EventCode::kCharacterLevelUpdated);
}

void Self::setHwanLevel(uint8_t hwanLevel) {
  hwanLevel_ = hwanLevel;
}

void Self::setSkillPoints(uint32_t skillPoints) {
  skillPoints_ = skillPoints;
  eventBroker_.publishEvent(event::EventCode::kCharacterSkillPointsUpdated);
}

void Self::setAvailableStatPoints(uint16_t statPoints) {
  availableStatPoints_ = statPoints;
  eventBroker_.publishEvent(event::EventCode::kCharacterAvailableStatPointsUpdated);
}

void Self::setCurrentExpAndSpExp(uint64_t currentExperience, uint64_t currentSpExperience) {
  currentExperience_ = currentExperience;
  currentSpExperience_ = currentSpExperience;
  eventBroker_.publishEvent(event::EventCode::kCharacterExperienceUpdated);
}

void Self::setHwanSpeed(float hwanSpeed) {
  hwanSpeed_ = hwanSpeed;
}

void Self::setBodyState(packet::enums::BodyState bodyState) {
  bodyState_ = bodyState;
  eventBroker_.publishEvent<event::EntityBodyStateChanged>(globalId);
}

void Self::setHwanPoints(uint8_t hwanPoints) {
  hwanPoints_ = hwanPoints;
  eventBroker_.publishEvent(event::EventCode::kHwanPointsUpdated);
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
  sro::Position intersectionPos(currentPosition.regionId(), static_cast<float>(intersectionPoint->x()), 0.0f, static_cast<float>(intersectionPoint->y()));
  // Start timer
  const auto seconds = helpers::secondsToTravel(currentPosition, intersectionPos, currentSpeed());
  enteredNewRegionEventId_ = eventBroker.publishDelayedEvent(std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)), event::EventCode::kEnteredNewRegion);
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

void Self::setCurrentMp(uint32_t mp) {
  mp_ = mp;
  eventBroker_.publishEvent(event::EventCode::kMpChanged);
}

void Self::setMaxHpMp(uint32_t maxHp, uint32_t maxMp) {
  const bool changed = optionalValueIsDifferentFromNewValue(maxHp_, maxHp) ||
                       optionalValueIsDifferentFromNewValue(maxMp_, maxMp);
  maxHp_ = maxHp;
  maxMp_ = maxMp;
  if (changed) {
    eventBroker_.publishEvent(event::EventCode::kMaxHpMpChanged);
  }
}

void Self::setStatPoints(uint16_t strPoints, uint16_t intPoints) {
  const bool changed = optionalValueIsDifferentFromNewValue(strPoints_, strPoints) ||
                       optionalValueIsDifferentFromNewValue(intPoints_, intPoints);
  const int pointsUsed = (strPoints_ ? (strPoints - *strPoints_) : 0) +
                         (intPoints_ ? (intPoints - *intPoints_) : 0);
  strPoints_ = strPoints;
  intPoints_ = intPoints;
  if (pointsUsed > 0) {
    // TODO: This packet is received when stat points are added as well as when equipment is changed. In the first case, we should reduce our available stat points. In the second case, we should not.
    if (availableStatPoints_ < pointsUsed) {
      throw std::runtime_error("Used more points than available stat points");
    }
    availableStatPoints_ -= pointsUsed;
    eventBroker_.publishEvent(event::EventCode::kCharacterAvailableStatPointsUpdated);
  }
  if (changed) {
    eventBroker_.publishEvent(event::EventCode::kStatsChanged);
  }
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
  for (const auto &skill : skills_) {
    if (!skill.enabled) {
      LOG(WARNING) << absl::StreamFormat("Received a skill (%d) which is not enabled", skill.id);
    }
  }
}

void Self::learnSkill(sro::scalar_types::ReferenceSkillId skillId) {
  // Check if any of our current skills have the same group ID.
  const auto &newSkillData = gameData_.skillData().getSkillById(skillId);
  bool foundSkill = false;
  for (auto &skill : skills_) {
    const auto &existingSkillData = gameData_.skillData().getSkillById(skill.id);
    if (newSkillData.groupId == existingSkillData.groupId) {
      // Overwrite this skill.
      eventBroker_.publishEvent<event::LearnSkillSuccess>(skillId, skill.id);
      skill.id = skillId;
      foundSkill = true;
      break;
    }
  }
  if (!foundSkill) {
    // Did not find existing skill with this group ID, create new.
    skills_.emplace_back(skillId, /*enabled=*/true);
    eventBroker_.publishEvent<event::LearnSkillSuccess>(skillId);
  }
}

void Self::learnMastery(sro::scalar_types::ReferenceMasteryId masteryId, uint8_t masteryLevel) {
  bool foundMastery = false;
  for (auto &mastery : masteries_) {
    if (mastery.id == masteryId) {
      VLOG(1) << absl::StreamFormat("Found mastery (ID:%d). Updating level from %d to %d", masteryId, mastery.level, masteryLevel);
      foundMastery = true;
      mastery.level = masteryLevel;
      break;
    }
  }
  if (!foundMastery) {
    VLOG(1) << absl::StreamFormat("Did not find mastery %d. Creating new with level %d", masteryId, masteryLevel);
    if (masteryLevel != 1) {
      LOG(WARNING) << absl::StreamFormat("Super weird that we learned a new mastery and the level isn't 1. It's %d", masteryLevel);
    }
    masteries_.emplace_back(masteryId, masteryLevel);
  }
  eventBroker_.publishEvent<event::LearnMasterySuccess>(masteryId);
}

void Self::setGold(uint64_t goldAmount) {
  gold_ = goldAmount;
  eventBroker_.publishEvent(event::EventCode::kInventoryGoldUpdated);
}

void Self::setStorageGold(uint64_t goldAmount) {
  storageGold_ = goldAmount;
  eventBroker_.publishEvent(event::EventCode::kStorageGoldUpdated);
}

void Self::setGuildStorageGold(uint64_t goldAmount) {
  guildStorageGold_ = goldAmount;
  eventBroker_.publishEvent(event::EventCode::kGuildStorageGoldUpdated);
}

void Self::usedAnItem(type_id::TypeId typeData, std::optional<std::chrono::milliseconds> cooldown, broker::EventBroker &eventBroker) {
  // For now, we figure out the cooldown duration right here. Maybe in the future, it should be passed into this function.
  if (itemCooldownEventIdMap_.find(typeData) != itemCooldownEventIdMap_.end()) {
    // This should never happen if all cooldowns are accurate.
    //  However, in the case of purification pills, which are maybe the only item with no real cooldown, if we create an artificial cooldown in the bot and use multiple through the client, this can trigger.
    // throw std::runtime_error("Used an item (" + type_id::toString(typeData) + "), but it's already on cooldown");
    LOG(INFO) << "Used an item (" << type_id::toString(typeData) << "), but it's already on cooldown";
    return;
  }

  if (!cooldown) {
    LOG(INFO) << "Used an item (" << type_id::toString(typeData) << "), but we don't know its cooldown time.";
    return;
  }

  // Publish a delayed event for when the cooldown ends.
  const auto itemCooldownDelayedEventId = eventBroker.publishDelayedEvent<event::ItemCooldownEnded>(*cooldown, typeData);
  itemCooldownEventIdMap_.emplace(typeData, itemCooldownDelayedEventId);
}

void Self::itemCooldownEnded(type_id::TypeId itemTypeData) {
  auto it = itemCooldownEventIdMap_.find(itemTypeData);
  if (it != itemCooldownEventIdMap_.end()) {
    itemCooldownEventIdMap_.erase(it);
  } else {
    LOG(INFO) << "Item cooldown ended, but we're not tracking it!?";
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

uint8_t Self::hwanLevel() const {
  return hwanLevel_;
}

uint32_t Self::getSkillPoints() const {
  return skillPoints_;
}

uint16_t Self::getAvailableStatPoints() const {
  return availableStatPoints_;
}

uint64_t Self::getCurrentExperience() const {
  return currentExperience_;
}

uint64_t Self::getCurrentSpExperience() const {
  return currentSpExperience_;
}

int Self::getHpPotionDelay() const {
  const bool havePanic = (modernStateLevels_[helpers::toBitNum(packet::enums::AbnormalStateFlag::kPanic)] > 0);
  return potionDelayMs_ + (havePanic ? kPanicPotionDelayIncreaseMs_ : 0);
}

int Self::getMpPotionDelay() const {
  const bool haveCombustion = (modernStateLevels_[helpers::toBitNum(packet::enums::AbnormalStateFlag::kCombustion)] > 0);
  return potionDelayMs_ + (haveCombustion ? kCombustionPotionDelayIncreaseMs_ : 0);
}

int Self::getVigorPotionDelay() const {
  // Vigors are not affected by panic or combustion.
  return potionDelayMs_;
}

int Self::getHpGrainDelay() const {
  const bool havePanic = (modernStateLevels_[helpers::toBitNum(packet::enums::AbnormalStateFlag::kPanic)] > 0);
  return kGrainDelayMs_ + (havePanic ? kPanicPotionDelayIncreaseMs_ : 0);
}

int Self::getMpGrainDelay() const {
  const bool haveCombustion = (modernStateLevels_[helpers::toBitNum(packet::enums::AbnormalStateFlag::kCombustion)] > 0);
  return kGrainDelayMs_ + (haveCombustion ? kCombustionPotionDelayIncreaseMs_ : 0);
}

int Self::getVigorGrainDelay() const {
  // Vigors are not affected by panic or combustion.
  return kGrainDelayMs_;
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

uint8_t Self::hwanPoints() const {
  return hwanPoints_;
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

std::optional<uint16_t> Self::strPoints() const {
  return strPoints_;
}

std::optional<uint16_t> Self::intPoints() const {
  return intPoints_;
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

bool Self::haveSkill(sro::scalar_types::ReferenceObjectId id) const {
  for (const auto &i : skills_) {
    if (i.id == id) {
      return i.enabled;
    }
  }
  return false;
}

uint8_t Self::getMasteryLevel(sro::scalar_types::ReferenceMasteryId id) const {
  for (const packet::structures::Mastery &mastery : masteries_) {
    if (mastery.id == id) {
      return mastery.level;
    }
  }
  LOG(WARNING) << "Asking for mastery level which we're not tracking";
  return 0;
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

bool Self::canUseItem(type_id::TypeId itemTypeId) const {
  return canUseItem(type_id::TypeCategory(itemTypeId));
}

bool Self::canUseItem(type_id::TypeCategory itemType) const {
  if (lifeState == sro::entity::LifeState::kDead) {
    // When dead, what types of items can we use?
    if (type_id::categories::kResurrection.contains(itemType)) {
      // This is ok.
    } else {
      // So far, the only item I know that we can use while dead is a resurrection scroll.
      return false;
    }
  } else if (lifeState == sro::entity::LifeState::kGone) {
    // Cannot use anything while unspawned
    return false;
  }
  // Other states are Alive or Embryo. Alive is obvious. Embryo seems to be the life state given to an entity upon first spawning into the world, but it never gets changed to Alive. Until we have good reason to do otherwise, we'll treat Embryo as Alive.
  const bool itemIsOnCooldown = (itemCooldownEventIdMap_.find(itemType.getTypeId()) != itemCooldownEventIdMap_.end());
  if (itemIsOnCooldown) {
    return false;
  }
  // TODO: Other reasons?
  return true;
}

// =====================================Packets-in-flight state=====================================
// Getters
bool Self::haveUserPurchaseRequest() const {
  return userPurchaseRequest_.has_value();
}

packet::structures::ItemMovement Self::getUserPurchaseRequest() const {
  if (!userPurchaseRequest_.has_value()) {
    throw std::runtime_error("Self: Trying to get user purchase request that does not exist");
  }
  return userPurchaseRequest_.value();
}

// Setters
void Self::setUserPurchaseRequest(const packet::structures::ItemMovement &itemMovement) {
  userPurchaseRequest_ = itemMovement;
}

void Self::resetUserPurchaseRequest() {
  userPurchaseRequest_.reset();
}

// =================================================================================================

void Self::setTrainingAreaGeometry(std::unique_ptr<entity::Geometry> &&geometry) {
  trainingAreaGeometry = std::move(geometry);
  eventBroker_.publishEvent(event::EventCode::kTrainingAreaSet);
}

void Self::resetTrainingAreaGeometry() {
  trainingAreaGeometry.reset();
  eventBroker_.publishEvent(event::EventCode::kTrainingAreaReset);
}

// =================================================================================================

bool Self::inTown() const {
  const auto &ourRegion = gameData_.refRegion().getRegion(position().regionId());
  return !ourRegion.isBattleField;
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