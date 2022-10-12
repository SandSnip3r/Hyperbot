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
    // In order to be able to interpolate position in the future, we need to update these values
    position_ = interpolatedPosition;
    startedMovingTime = currentTime;

    if (destinationPosition) {
      if (!movingEventId) {
        throw std::runtime_error("We're moving towards some desitnation position, but there's no timer running");
      }
      // Update timer for new speed
      eventBroker.cancelDelayedEvent(*movingEventId);
      auto seconds = helpers::secondsToTravel(position_, *destinationPosition, privateCurrentSpeed());
      movingEventId = eventBroker.publishDelayedEvent(std::make_unique<event::EntityMovementTimerEnded>(globalId), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
    }

    // Publish a movement began event since this is essentially creating a new movement
    eventBroker.publishEvent(std::make_unique<event::EntityMovementBegan>(globalId));
  }
}

void MobileEntity::setAngle(sro::Angle angle, broker::EventBroker &eventBroker) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (moving()) {
    throw std::runtime_error("We're moving and changing our angle");
  }
  this->angle_ = angle;
  eventBroker.publishEvent(std::make_unique<event::EntityNotMovingAngleChanged>(globalId));
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
  const auto whereWeThoughtWeWere = interpolateCurrentPosition(currentTime);
  position_ = position;
  // TODO: Need angle?
  if (moving()) {
    startedMovingTime = currentTime;
    const auto offByDistance = sro::position_math::calculateDistance2D(whereWeThoughtWeWere, position);
    // Might be worth recalculating travel time and starting a new timer
    // TODO: Should we fire an event. This update might be worth sending to the UI at least
  }
  eventBroker.publishEvent(std::make_unique<event::EntitySyncedPosition>(globalId));
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
  eventBroker.publishEvent(std::make_unique<event::EntityMovementEnded>(globalId));
}

void MobileEntity::cancelMovement(broker::EventBroker &eventBroker) {
  if (movingEventId) {
    eventBroker.cancelDelayedEvent(*movingEventId);
    movingEventId.reset();
  }
  moving_ = false;
  destinationPosition.reset();
}

sro::Position MobileEntity::interpolateCurrentPosition(const std::chrono::high_resolution_clock::time_point &currentTime) const {
  if (!moving()) {
    return position_;
  }
  auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime-startedMovingTime).count();
  if (destinationPosition) {
    auto totalDistance = sro::position_math::calculateDistance2D(position_, *destinationPosition);
    if (totalDistance < 0.0001 /* TODO: Use a double equal function */) {
      // We're at our destination
      return *destinationPosition;
    }
    auto expectedTravelTimeSeconds = totalDistance / privateCurrentSpeed();
    double percentTraveled = (elapsedTimeMs/1000.0) / expectedTravelTimeSeconds;
    if (percentTraveled < 0) {
      throw std::runtime_error("Self: Traveled negative distance");
    } else if (percentTraveled == 0) {
      return position_;
    } else if (percentTraveled == 1) {
      // TODO: I think this case should actually just be >=1
      //  It doesnt make much sense to travel past our destination position, that just means our timer is off, i think
      return *destinationPosition;
    } else {
      const auto resultPos = sro::position_math::interpolateBetweenPoints(position_, *destinationPosition, percentTraveled);
      if (percentTraveled > 1) {
        LOG() << "Traveled past destination (" << percentTraveled << ")" << std::endl;
        if (percentTraveled == std::numeric_limits<double>::infinity()) {
          throw std::runtime_error("Nooo");
        }
      }
      return resultPos;
    }
  } else {
    float angleRadians = angle_/static_cast<float>(std::numeric_limits<std::remove_reference_t<decltype(angle_)>>::max()) * sro::constants::k2Pi;
    float xOffset = std::cos(angleRadians) * privateCurrentSpeed() * (elapsedTimeMs/1000.0);
    float zOffset = std::sin(angleRadians) * privateCurrentSpeed() * (elapsedTimeMs/1000.0);
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
  eventBroker.publishEvent(std::make_unique<event::EntityMovementEnded>(globalId));
  if (angleUpdated) {
    eventBroker.publishEvent(std::make_unique<event::EntityNotMovingAngleChanged>(globalId));
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
  eventBroker.publishEvent(std::make_unique<event::EntityMovementBegan>(globalId));
  movingEventId = eventBroker.publishDelayedEvent(std::make_unique<event::EntityMovementTimerEnded>(globalId), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
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

  eventBroker.publishEvent(std::make_unique<event::EntityMovementBegan>(globalId));
}

void MobileEntity::initializeAsMoving(const sro::Position &destinationPosition) {
  this->moving_ = true;
  this->startedMovingTime = std::chrono::high_resolution_clock::now();
  this->destinationPosition = destinationPosition;
}

void MobileEntity::initializeAsMoving(sro::Angle destinationAngle) {
  this->moving_ = true;
  this->startedMovingTime = std::chrono::high_resolution_clock::now();
  this->angle_ = destinationAngle;
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
    eventBroker.publishEvent(std::make_unique<event::EntityLifeStateChanged>(globalId));
  }
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