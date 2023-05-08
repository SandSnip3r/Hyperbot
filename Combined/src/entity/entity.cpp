#include "entity.hpp"
#include "helpers.hpp"
#include "logging.hpp"

#include <silkroad_lib/constants.h>
#include <silkroad_lib/position_math.h>

namespace entity {

void Entity::initializePosition(const sro::Position &position) {
  position_ = position;
}

void Entity::initializeAngle(sro::Angle angle) {
  angle_ = angle;
}

sro::Position Entity::position() const {
  return position_;
}

sro::Angle Entity::angle() const {
  return angle_;
}

EntityType Entity::entityType() const  {
  if (dynamic_cast<const Character*>(this)) {
    if (dynamic_cast<const PlayerCharacter*>(this)) {
      return EntityType::kPlayerCharacter;
    } else if (dynamic_cast<const NonplayerCharacter*>(this)) {
      if (dynamic_cast<const Monster*>(this)) {
        return EntityType::kMonster;
      } else {
        return EntityType::kNonplayerCharacter;
      }
    } else {
      return EntityType::kCharacter;
    }
  } else if (dynamic_cast<const Item*>(this)) {
    return EntityType::kItem;
  } else if (dynamic_cast<const Portal*>(this)) {
    return EntityType::kPortal;
  }
  throw std::runtime_error("Cannot get entity type");
}

void MobileEntity::initializeAsMoving(const sro::Position &destinationPosition) {
  std::lock_guard<std::mutex> lock(mutex_);
  this->moving_ = true;
  this->startedMovingTime = std::chrono::high_resolution_clock::now();
  this->destinationPosition = destinationPosition;
}

void MobileEntity::initializeAsMoving(sro::Angle destinationAngle) {
  std::lock_guard<std::mutex> lock(mutex_);
  this->moving_ = true;
  this->startedMovingTime = std::chrono::high_resolution_clock::now();
  this->angle_ = destinationAngle;
}

void MobileEntity::registerGeometryBoundary(std::unique_ptr<Geometry> geometry, broker::EventBroker &eventBroker) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (geometry_) {
    throw std::runtime_error("MobileEntity already holds geometry");
  }
  geometry_ = std::move(geometry);
  if (moving()) {
    checkIfWillCrossGeometryBoundary(eventBroker);
  }
}

void MobileEntity::resetGeometryBoundary(broker::EventBroker &eventBroker) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (geometry_) {
    geometry_.reset();
  }
  cancelGeometryEvents(eventBroker);
}

void MobileEntity::cancelEvents(broker::EventBroker &eventBroker) {
  std::lock_guard<std::mutex> lock(mutex_);
  privateCancelEvents(eventBroker);
}

bool MobileEntity::moving() const {
  return moving_;
}

sro::Position MobileEntity::position() const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto currentTime = std::chrono::high_resolution_clock::now();
  return interpolateCurrentPosition(currentTime);
}

float MobileEntity::currentSpeed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return privateCurrentSpeed();
}

sro::Position MobileEntity::positionAfterTime(float seconds) const {
  const auto futureTime = std::chrono::high_resolution_clock::now() + std::chrono::microseconds(static_cast<std::chrono::microseconds::rep>(seconds * 1'000'000));
  return interpolateCurrentPosition(futureTime);
}

void MobileEntity::setSpeed(float walkSpeed, float runSpeed, broker::EventBroker &eventBroker) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  if (walkSpeed == this->walkSpeed && runSpeed == this->runSpeed) {
    // Didnt actually change
    return;
  }
  // Get interpolated position before change speed, since that calulation depends on our current speed
  const auto interpolatedPosition = interpolateCurrentPosition(currentTime);
  this->walkSpeed = walkSpeed;
  this->runSpeed = runSpeed;
  if (moving()) {
    if (destinationPosition) {
      privateSetMovingToDestination(interpolatedPosition, *destinationPosition, eventBroker);
    } else {
      privateSetMovingTowardAngle(interpolatedPosition, angle_, eventBroker);
    }
  }
}

void MobileEntity::setAngle(sro::Angle angle, broker::EventBroker &eventBroker) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (moving()) {
    throw std::runtime_error("We're moving and changing our angle");
  }
  this->angle_ = angle;
  eventBroker.publishEvent<event::EntityNotMovingAngleChanged>(globalId);
}

void MobileEntity::setMotionState(entity::MotionState motionState, broker::EventBroker &eventBroker) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  auto prevSpeed = privateCurrentSpeed();
  bool changedSpeed{false};
  if (this->lastMotionState && *this->lastMotionState == entity::MotionState::kWalk && motionState == entity::MotionState::kRun) {
    // Entity changed from walking to running
    changedSpeed = true;
  } else if (this->lastMotionState && *this->lastMotionState == entity::MotionState::kRun && motionState == entity::MotionState::kWalk) {
    // Entity changed from running to walking
    changedSpeed = true;
  }

  std::optional<sro::Position> srcPosition;
  if (changedSpeed) {
    // Since we changed speed, figure out where we currently are
    srcPosition = interpolateCurrentPosition(currentTime);
  }

  // Update our motion state
  this->motionState = motionState;
  if (this->motionState == entity::MotionState::kRun || this->motionState == entity::MotionState::kWalk) {
    // Save whether we were walking or running last
    lastMotionState = this->motionState;
  }

  if (changedSpeed && moving()) {
    if (!srcPosition) {
      throw std::runtime_error("Changes speed, but dont know our position when it happened");
    }
    if (destinationPosition) {
      privateSetMovingToDestination(srcPosition, *destinationPosition, eventBroker);
    } else {
      privateSetMovingTowardAngle(srcPosition, angle_, eventBroker);
    }
  }
}

void MobileEntity::setStationaryAtPosition(const sro::Position &position, broker::EventBroker &eventBroker) {
  std::lock_guard<std::mutex> lock(mutex_);
  privateSetStationaryAtPosition(position, eventBroker);
}

void MobileEntity::syncPosition(const sro::Position &position, broker::EventBroker &eventBroker) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  if (moving()) {
    if (destinationPosition) {
      privateSetMovingToDestination(position, *destinationPosition, eventBroker);
    } else {
      privateSetMovingTowardAngle(position, angle_, eventBroker);
    }
  } else {
    position_ = position;
    eventBroker.publishEvent<event::EntityPositionUpdated>(globalId);
  }
}

void MobileEntity::setMovingToDestination(const std::optional<sro::Position> &sourcePosition, const sro::Position &destinationPosition, broker::EventBroker &eventBroker) {
  std::lock_guard<std::mutex> lock(mutex_);
  privateSetMovingToDestination(sourcePosition, destinationPosition, eventBroker);
}

void MobileEntity::setMovingTowardAngle(const std::optional<sro::Position> &sourcePosition, const sro::Angle angle, broker::EventBroker &eventBroker) {
  std::lock_guard<std::mutex> lock(mutex_);
  privateSetMovingTowardAngle(sourcePosition, angle, eventBroker);
}

void MobileEntity::movementTimerCompleted(broker::EventBroker &eventBroker) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!movingEventId) {
    throw std::runtime_error("MobileEntity: Movement timer completed, but had no running timer");
  }
  if (!moving()) {
    throw std::runtime_error("MobileEntity: Movement timer completed, but entity wasnt moving");
  }
  if (!destinationPosition) {
    throw std::runtime_error("MobileEntity: Movement timer completed, but we dont know where the entity was going");
  }
  position_ = *destinationPosition;
  cancelMovement(eventBroker);
  eventBroker.publishEvent<event::EntityMovementEnded>(globalId);
}

void MobileEntity::privateCancelEvents(broker::EventBroker &eventBroker) {
  if (movingEventId) {
    eventBroker.cancelDelayedEvent(*movingEventId);
    movingEventId.reset();
  }
  cancelGeometryEvents(eventBroker);
}

void MobileEntity::cancelMovement(broker::EventBroker &eventBroker) {
  privateCancelEvents(eventBroker);
  moving_ = false;
  destinationPosition.reset();
}

sro::Position MobileEntity::interpolateCurrentPosition(const std::chrono::high_resolution_clock::time_point &currentTime) const {
  if (!moving()) {
    return position_;
  }
  const auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime-startedMovingTime).count();
  if (destinationPosition) {
    auto totalDistance = sro::position_math::calculateDistance2d(position_, *destinationPosition);
    if (totalDistance < 0.0001 /* TODO: Use a double equal function */) {
      // We're at our destination
      return *destinationPosition;
    }
    const auto expectedTravelTimeSeconds = totalDistance / privateCurrentSpeed();
    const double percentTraveled = (elapsedTimeMs/1000.0) / expectedTravelTimeSeconds;
    if (percentTraveled < 0) {
      throw std::runtime_error("Self: Traveled negative distance");
    } else if (percentTraveled == 0) {
      return position_;
    } else if (percentTraveled >= 1) {
      //  It doesnt make much sense to travel past our destination position, that just means our timer is off. We'll truncate to the destination position.
      if (percentTraveled == std::numeric_limits<double>::infinity()) {
        throw std::runtime_error("Traveled an infinite distance.");
      }
      return *destinationPosition;
    } else {
      return sro::position_math::interpolateBetweenPoints(position_, *destinationPosition, percentTraveled);
    }
  } else {
    const float angleRadians = angle_/static_cast<float>(std::numeric_limits<std::remove_reference_t<decltype(angle_)>>::max()) * sro::constants::k2Pi;
    const float xOffset = std::cos(angleRadians) * privateCurrentSpeed() * (elapsedTimeMs/1000.0);
    const float zOffset = std::sin(angleRadians) * privateCurrentSpeed() * (elapsedTimeMs/1000.0);
    return sro::position_math::createNewPositionWith2dOffset(position_, xOffset, zOffset);
  }
}

float MobileEntity::privateCurrentSpeed() const {
  if (motionState == MotionState::kRun) {
    return runSpeed;
  } else if (motionState == MotionState::kWalk) {
    return walkSpeed;
  } else if (motionState == MotionState::kStand) {
    if (lastMotionState) {
      if (*lastMotionState == MotionState::kRun) {
        return runSpeed;
      } else if (*lastMotionState == MotionState::kWalk) {
        return walkSpeed;
      } else {
        throw std::runtime_error("Motion state is Stand, last motion state isnt walk or run ("+std::to_string(static_cast<int>(*lastMotionState))+")");
      }
    } else {
      // MotionState: Stand, no previous motion state assuming that we're running
      return runSpeed;
    }
  } else {
    // TODO: Understand what the other cases are here
    // TODO: Include zerk
    throw std::runtime_error("Trying to get speed, but not walking nor running");
  }
}

void MobileEntity::privateSetStationaryAtPosition(const sro::Position &position, broker::EventBroker &eventBroker) {
  cancelMovement(eventBroker);
  bool angleUpdated{false};
  if (position_ != position) {
    // Calculate angle of line created by these two points. Entity is facing that dirction
    angle_ = sro::position_math::calculateAngleOfLine(position_, position);
    angleUpdated = true;
  }
  position_ = position;
  eventBroker.publishEvent<event::EntityMovementEnded>(globalId);
  if (angleUpdated) {
    eventBroker.publishEvent<event::EntityNotMovingAngleChanged>(globalId);
  }
}

void MobileEntity::privateSetMovingToDestination(const std::optional<sro::Position> &sourcePosition, const sro::Position &destinationPosition, broker::EventBroker &eventBroker) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  if (sourcePosition) {
    position_ = *sourcePosition;
  } else if (moving()) {
    // We've pivoted while moving, calculate where we are and save that
    position_ = interpolateCurrentPosition(currentTime);
  }
  if (position_ == destinationPosition) {
    // Not going anywhere
    privateSetStationaryAtPosition(destinationPosition, eventBroker);
    return;
  }
  cancelMovement(eventBroker);
  moving_ = true;
  startedMovingTime = currentTime;
  this->destinationPosition = destinationPosition;

  // Start timer
  const auto seconds = helpers::secondsToTravel(position_, *this->destinationPosition, privateCurrentSpeed());
  eventBroker.publishEvent<event::EntityMovementBegan>(globalId);
  movingEventId = eventBroker.publishDelayedEvent<event::EntityMovementTimerEnded>(std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)), globalId);
  checkIfWillCrossGeometryBoundary(eventBroker);
}

void MobileEntity::privateSetMovingTowardAngle(const std::optional<sro::Position> &sourcePosition, const sro::Angle angle, broker::EventBroker &eventBroker) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  if (sourcePosition) {
    position_ = *sourcePosition;
  } else if (moving()) {
    // We've pivoted while moving, calculate where we are and save that
    position_ = interpolateCurrentPosition(currentTime);
  }
  cancelMovement(eventBroker);
  moving_ = true;
  startedMovingTime = currentTime;
  this->angle_ = angle;

  eventBroker.publishEvent<event::EntityMovementBegan>(globalId);
  checkIfWillCrossGeometryBoundary(eventBroker);
}

void MobileEntity::checkIfWillCrossGeometryBoundary(broker::EventBroker &eventBroker) {
  if (!geometry_) {
    // No geometry to collide with
    return;
  }
  const auto currentTime = std::chrono::high_resolution_clock::now();
  const auto currentPosition = interpolateCurrentPosition(currentTime);

  if (destinationPosition) {
    // Moving to some destination
    auto maybeTimeUntilEnter = geometry_->timeUntilEnter(currentPosition, *destinationPosition, privateCurrentSpeed());
    if (maybeTimeUntilEnter) {
      LOG() << " Entity will enter the geometry boundary in " << *maybeTimeUntilEnter << " second(s)" << std::endl;
      // TODO: Need some way to reference the geometry from the event
      enterGeometryEventId_ = eventBroker.publishDelayedEvent<event::EntityEnteredGeometry>(std::chrono::milliseconds(static_cast<uint64_t>((*maybeTimeUntilEnter)*1000)), globalId);
    }
    auto maybeTimeUntilExit = geometry_->timeUntilExit(currentPosition, *destinationPosition, privateCurrentSpeed());
    if (maybeTimeUntilExit) {
      LOG() << " Entity will exit the geometry boundary in " << *maybeTimeUntilExit << " second(s)" << std::endl;
      // TODO: Need some way to reference the geometry from the event
      exitGeometryEventId_ = eventBroker.publishDelayedEvent<event::EntityExitedGeometry>(std::chrono::milliseconds(static_cast<uint64_t>((*maybeTimeUntilExit)*1000)), globalId);
    }
  } else {
    // Moving towards some angle
    auto maybeTimeUntilEnter = geometry_->timeUntilEnter(currentPosition, angle_, privateCurrentSpeed());
    if (maybeTimeUntilEnter) {
      // TODO: Need some way to reference the geometry from the event
      enterGeometryEventId_ = eventBroker.publishDelayedEvent<event::EntityEnteredGeometry>(std::chrono::milliseconds(static_cast<uint64_t>((*maybeTimeUntilEnter)*1000)), globalId);
    }
    auto maybeTimeUntilExit = geometry_->timeUntilExit(currentPosition, angle_, privateCurrentSpeed());
    if (maybeTimeUntilExit) {
      // TODO: Need some way to reference the geometry from the event
      exitGeometryEventId_ = eventBroker.publishDelayedEvent<event::EntityExitedGeometry>(std::chrono::milliseconds(static_cast<uint64_t>((*maybeTimeUntilExit)*1000)), globalId);
    }
  }
}

void MobileEntity::cancelGeometryEvents(broker::EventBroker &eventBroker) {
  if (enterGeometryEventId_) {
    eventBroker.cancelDelayedEvent(*enterGeometryEventId_);
    enterGeometryEventId_.reset();
  }
  if (exitGeometryEventId_) {
    eventBroker.cancelDelayedEvent(*exitGeometryEventId_);
    exitGeometryEventId_.reset();
  }
}

// ============================================================================================================================================

void Character::setLifeState(sro::entity::LifeState newLifeState, broker::EventBroker &eventBroker) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  const bool changed = lifeState != newLifeState;
  lifeState = newLifeState;
  if (newLifeState == sro::entity::LifeState::kDead) {
    privateSetStationaryAtPosition(interpolateCurrentPosition(currentTime), eventBroker);
  }
  if (changed) {
    eventBroker.publishEvent<event::EntityLifeStateChanged>(globalId);
  }
}

bool Character::knowCurrentHp() const {
  return currentHp_.has_value();
}

uint32_t Character::currentHp() const {
  if (!currentHp_) {
    throw std::runtime_error("Trying to get character's unknown hp");
  }
  return *currentHp_;
}

void Character::setCurrentHp(uint32_t hp, broker::EventBroker &eventBroker) {
  currentHp_ = hp;
  eventBroker.publishEvent<event::EntityHpChanged>(globalId);
}

// ============================================================================================================================================

void PlayerCharacter::addBuff(sro::scalar_types::ReferenceObjectId skillRefId, broker::EventBroker &eventBroker) {
  buffs.emplace(skillRefId);
  eventBroker.publishEvent<event::BuffAdded>(globalId, skillRefId);
}

void PlayerCharacter::removeBuff(sro::scalar_types::ReferenceObjectId skillRefId, broker::EventBroker &eventBroker) {
  auto buffIt = buffs.find(skillRefId);
  if (buffIt == buffs.end()) {
    throw std::runtime_error("Trying to remove buff from entity, but cannot find it");
  }
  buffs.erase(buffIt);
  eventBroker.publishEvent<event::BuffRemoved>(globalId, skillRefId);
}

// ============================================================================================================================================

void Item::removeOwnership(broker::EventBroker &eventBroker) {
  ownerJId.reset();
  eventBroker.publishEvent<event::EntityOwnershipRemoved>(globalId);
}

// ============================================================================================================================================

uint32_t Monster::getMaxHp(const pk2::CharacterData &characterData) const {
  if (!characterData.haveCharacterWithId(refObjId)) {
    throw std::runtime_error("Don't have character data to get max HP");
  }
  const auto &data = characterData.getCharacterById(refObjId);
  uint32_t hp = data.maxHp;
  using RarityRawType = std::underlying_type_t<decltype(rarity)>;
  const auto partylessRarity = static_cast<sro::entity::MonsterRarity>(static_cast<RarityRawType>(rarity) & (static_cast<RarityRawType>(sro::entity::MonsterRarity::kPartyFlag)-1));
  switch (partylessRarity) {
    case sro::entity::MonsterRarity::kChampion:
      hp *= 2;
      break;
    case sro::entity::MonsterRarity::kGiant:
      hp *= 20;
      break;
    case sro::entity::MonsterRarity::kElite:
      hp *= 30;
      break;
    default:
      LOG() << "Asking for max HP of an unknown monster rarity" << std::endl;
      break;
  }
  if (flags::isSet(rarity, sro::entity::MonsterRarity::kPartyFlag)) {
    // Party monsters have a flat x10 across the base
    hp *= 10;
  }
  return hp;
}

// ============================================================================================================================================

std::ostream& operator<<(std::ostream &stream, MotionState motionState) {
  switch(motionState) {
    case MotionState::kStand:
      stream << "Stand";
      break;
    case MotionState::kSkill:
      stream << "Skill";
      break;
    case MotionState::kWalk:
      stream << "Walk";
      break;
    case MotionState::kRun:
      stream << "Run";
      break;
    case MotionState::kSit:
      stream << "Sit";
      break;
    default:
      stream << "UNKNOWN";
      break;
  }
  return stream;
}

} // namespace entity