#include "helpers.hpp"
#include "mobileEntity.hpp"

#include <silkroad_lib/position_math.hpp>
#include <silkroad_lib/constants.hpp>

#include <absl/log/log.h>

#include <limits>
#include <stdexcept>

namespace entity {

MobileEntity::~MobileEntity() {
  cancelEvents();
  if (movementTimerEndedSubscription_) {
    if (!eventBroker_) {
      throw std::runtime_error("Deconstructing MobileEntity; have open subscription but no event broker");
    }
    eventBroker_->unsubscribeFromEvent(*movementTimerEndedSubscription_);
  }
}

void MobileEntity::initializeAsMoving(const sro::Position &destinationPosition) {
  // std::unique_lock<std::mutex> lock(mutex_);
  this->moving_ = true;
  this->startedMovingTime = std::chrono::high_resolution_clock::now();
  this->destinationPosition = destinationPosition;
}

void MobileEntity::initializeAsMoving(sro::Angle destinationAngle) {
  // std::unique_lock<std::mutex> lock(mutex_);
  this->moving_ = true;
  this->startedMovingTime = std::chrono::high_resolution_clock::now();
  this->angle_ = destinationAngle;
}

void MobileEntity::initializeEventBroker(broker::EventBroker &eventBroker) {
  Entity::initializeEventBroker(eventBroker);
  if (moving()) {
    checkIfWillCrossGeometryBoundary();
  }
  // TODO: If we're moving to a destination, start a timer for arrival.

  if (movementTimerEndedSubscription_) {
    throw std::runtime_error("MobileEntity is already subscribed to events. Trying to subscribe again.");
  }
  auto handleFunction = std::bind(&MobileEntity::handleEvent, this, std::placeholders::_1);
  movementTimerEndedSubscription_ = eventBroker.subscribeToEvent(event::EventCode::kEntityMovementTimerEnded, handleFunction);
}

void MobileEntity::handleEvent(const event::Event *event) {
  if (event == nullptr) {
    throw std::runtime_error("MobileEntity::handleEvent given null event");
  }
  try {
    if (const auto *movementTimerEndedEvent = dynamic_cast<const event::EntityMovementTimerEnded*>(event); movementTimerEndedEvent != nullptr) {
      if (movementTimerEndedEvent->globalId == globalId) {
        movementTimerCompleted();
      }
    } else {
      LOG(WARNING) << "Unhandled event received: " << event::toString(event->eventCode);
    }
  } catch (std::exception &ex) {
    LOG(ERROR) << absl::StreamFormat("Error while handling event: \"%s\"", ex.what());
  }
}

void MobileEntity::registerGeometryBoundary(std::unique_ptr<Geometry> geometry) {
  // std::unique_lock<std::mutex> lock(mutex_);
  if (geometry_) {
    throw std::runtime_error("MobileEntity already holds geometry");
  }
  geometry_ = std::move(geometry);
  if (moving()) {
    checkIfWillCrossGeometryBoundary();
  }
}

void MobileEntity::resetGeometryBoundary() {
  // std::unique_lock<std::mutex> lock(mutex_);
  if (geometry_) {
    geometry_.reset();
  }
  cancelGeometryEvents();
}

void MobileEntity::cancelEvents() {
  // std::unique_lock<std::mutex> lock(mutex_);
  privateCancelEvents();
}

bool MobileEntity::moving() const {
  return moving_;
}

sro::Position MobileEntity::position() const {
  // std::unique_lock<std::mutex> lock(mutex_);
  const auto currentTime = std::chrono::high_resolution_clock::now();
  return interpolateCurrentPosition(currentTime);
}

float MobileEntity::currentSpeed() const {
  // std::unique_lock<std::mutex> lock(mutex_);
  return privateCurrentSpeed();
}

sro::Position MobileEntity::positionAfterTime(float seconds) const {
  const auto futureTime = std::chrono::high_resolution_clock::now() + std::chrono::microseconds(static_cast<std::chrono::microseconds::rep>(seconds * 1'000'000));
  return interpolateCurrentPosition(futureTime);
}

void MobileEntity::setSpeed(float walkSpeed, float runSpeed) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  // std::unique_lock<std::mutex> lock(mutex_);
  if (walkSpeed == this->walkSpeed && runSpeed == this->runSpeed) {
    // Didn't actually change
    return;
  }
  // Get interpolated position before change speed, since that calculation depends on our current speed
  const auto interpolatedPosition = interpolateCurrentPosition(currentTime);
  this->walkSpeed = walkSpeed;
  this->runSpeed = runSpeed;
  if (moving()) {
    if (destinationPosition) {
      privateSetMovingToDestination(interpolatedPosition, *destinationPosition);
    } else {
      privateSetMovingTowardAngle(interpolatedPosition, angle_);
    }
  }
}

void MobileEntity::setAngle(sro::Angle angle) {
  // std::unique_lock<std::mutex> lock(mutex_);
  if (moving()) {
    throw std::runtime_error("We're moving and changing our angle");
  }
  this->angle_ = angle;
  if (eventBroker_) {
    eventBroker_->publishEvent<event::EntityNotMovingAngleChanged>(globalId);
  } else {
    LOG(WARNING) << "Trying to publish angle changed event, but do not have event broker";
  }
}

void MobileEntity::setMotionState(entity::MotionState motionState) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  // std::unique_lock<std::mutex> lock(mutex_);
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
      privateSetMovingToDestination(srcPosition, *destinationPosition);
    } else {
      privateSetMovingTowardAngle(srcPosition, angle_);
    }
  }
}

void MobileEntity::setStationaryAtPosition(const sro::Position &position) {
  // std::unique_lock<std::mutex> lock(mutex_);
  privateSetStationaryAtPosition(position);
}

void MobileEntity::syncPosition(const sro::Position &position) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  // std::unique_lock<std::mutex> lock(mutex_);
  if (moving()) {
    if (destinationPosition) {
      privateSetMovingToDestination(position, *destinationPosition);
    } else {
      privateSetMovingTowardAngle(position, angle_);
    }
  } else {
    position_ = position;
    if (eventBroker_) {
      eventBroker_->publishEvent<event::EntityPositionUpdated>(globalId);
    } else {
      LOG(WARNING) << "Trying to publish position updated event, but do not have event broker";
    }
  }
}

void MobileEntity::setMovingToDestination(const std::optional<sro::Position> &sourcePosition, const sro::Position &destinationPosition) {
  // std::unique_lock<std::mutex> lock(mutex_);
  privateSetMovingToDestination(sourcePosition, destinationPosition);
}

void MobileEntity::setMovingTowardAngle(const std::optional<sro::Position> &sourcePosition, const sro::Angle angle) {
  // std::unique_lock<std::mutex> lock(mutex_);
  privateSetMovingTowardAngle(sourcePosition, angle);
}

void MobileEntity::movementTimerCompleted() {
  // std::unique_lock<std::mutex> lock(mutex_);
  if (!movingEventId) {
    throw std::runtime_error("MobileEntity: Movement timer completed, but had no running timer");
  }
  if (!moving()) {
    throw std::runtime_error("MobileEntity: Movement timer completed, but entity wasn't moving");
  }
  if (!destinationPosition) {
    throw std::runtime_error("MobileEntity: Movement timer completed, but we dont know where the entity was going");
  }
  position_ = *destinationPosition;
  movingEventId.reset();
  cancelMovement();
  if (eventBroker_) {
    eventBroker_->publishEvent<event::EntityMovementEnded>(globalId);
  } else {
    LOG(WARNING) << "Trying to publish movement ended event, but do not have event broker";
  }
}

void MobileEntity::privateCancelEvents() {
  if (movingEventId) {
    if (eventBroker_) {
      eventBroker_->cancelDelayedEvent(*movingEventId);
    } else {
      LOG(WARNING) << "Trying to cancel movement event, but do not have event broker";
    }
    movingEventId.reset();
  }
  cancelGeometryEvents();
}

void MobileEntity::cancelMovement() {
  privateCancelEvents();
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
      //  It doesn't make much sense to travel past our destination position, that just means our timer is off. We'll truncate to the destination position.
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
        throw std::runtime_error("Motion state is Stand, last motion state isn't walk or run ("+std::to_string(static_cast<int>(*lastMotionState))+")");
      }
    } else {
      // MotionState: Stand, no previous motion state assuming that we're running
      return runSpeed;
    }
  } else if (motionState == MotionState::kSit) {
    throw std::runtime_error("Trying to get current speed, but entity is sitting");
  } else {
    // TODO: Understand what the other cases are here
    // TODO: Include berserk
    throw std::runtime_error("Trying to get speed, but not walking nor running");
  }
}

void MobileEntity::privateSetStationaryAtPosition(const sro::Position &position) {
  cancelMovement();
  bool angleUpdated{false};
  if (position_ != position) {
    // Calculate angle of line created by these two points. Entity is facing that direction
    angle_ = sro::position_math::calculateAngleOfLine(position_, position);
    angleUpdated = true;
  }
  position_ = position;
  if (eventBroker_) {
    eventBroker_->publishEvent<event::EntityMovementEnded>(globalId);
    if (angleUpdated) {
      eventBroker_->publishEvent<event::EntityNotMovingAngleChanged>(globalId);
    }
  } else {
    LOG(WARNING) << "Trying to publish movement ended event, but do not have event broker";
  }
}

void MobileEntity::privateSetMovingToDestination(const std::optional<sro::Position> &sourcePosition, const sro::Position &destinationPosition) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  if (sourcePosition) {
    position_ = *sourcePosition;
  } else if (moving()) {
    // We've pivoted while moving, calculate where we are and save that
    position_ = interpolateCurrentPosition(currentTime);
  }
  if (position_ == destinationPosition) {
    // Not going anywhere
    privateSetStationaryAtPosition(destinationPosition);
    return;
  }
  cancelMovement();
  moving_ = true;
  startedMovingTime = currentTime;
  this->destinationPosition = destinationPosition;

  // Start timer
  if (eventBroker_) {
    const auto seconds = helpers::secondsToTravel(position_, *this->destinationPosition, privateCurrentSpeed());
    eventBroker_->publishEvent<event::EntityMovementBegan>(globalId);
    movingEventId = eventBroker_->publishDelayedEvent<event::EntityMovementTimerEnded>(std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)), globalId);
  } else {
    LOG(WARNING) << "Trying to publish movement began and movement timer ended event, but do not have event broker";
  }
  checkIfWillCrossGeometryBoundary();
}

void MobileEntity::privateSetMovingTowardAngle(const std::optional<sro::Position> &sourcePosition, const sro::Angle angle) {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  if (sourcePosition) {
    position_ = *sourcePosition;
  } else if (moving()) {
    // We've pivoted while moving, calculate where we are and save that
    position_ = interpolateCurrentPosition(currentTime);
  }
  cancelMovement();
  moving_ = true;
  startedMovingTime = currentTime;
  this->angle_ = angle;

  if (eventBroker_) {
    eventBroker_->publishEvent<event::EntityMovementBegan>(globalId);
  } else {
    LOG(WARNING) << "Trying to publish movement began event, but do not have event broker";
  }
  checkIfWillCrossGeometryBoundary();
}

void MobileEntity::checkIfWillCrossGeometryBoundary() {
  if (!eventBroker_) {
    LOG(WARNING) << "Trying to check if entity will cross geometry boundary, but do not have event broker";
    return;
  }
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
      // Entity will enter the geometry boundary in *maybeTimeUntilEnter second(s)
      // TODO: Need some way to reference the geometry from the event
      enterGeometryEventId_ = eventBroker_->publishDelayedEvent<event::EntityEnteredGeometry>(std::chrono::milliseconds(static_cast<uint64_t>((*maybeTimeUntilEnter)*1000)), globalId);
    }
    auto maybeTimeUntilExit = geometry_->timeUntilExit(currentPosition, *destinationPosition, privateCurrentSpeed());
    if (maybeTimeUntilExit) {
      // Entity will exit the geometry boundary in *maybeTimeUntilExit second(s)
      // TODO: Need some way to reference the geometry from the event
      exitGeometryEventId_ = eventBroker_->publishDelayedEvent<event::EntityExitedGeometry>(std::chrono::milliseconds(static_cast<uint64_t>((*maybeTimeUntilExit)*1000)), globalId);
    }
  } else {
    // Moving towards some angle
    auto maybeTimeUntilEnter = geometry_->timeUntilEnter(currentPosition, angle_, privateCurrentSpeed());
    if (maybeTimeUntilEnter) {
      // TODO: Need some way to reference the geometry from the event
      enterGeometryEventId_ = eventBroker_->publishDelayedEvent<event::EntityEnteredGeometry>(std::chrono::milliseconds(static_cast<uint64_t>((*maybeTimeUntilEnter)*1000)), globalId);
    }
    auto maybeTimeUntilExit = geometry_->timeUntilExit(currentPosition, angle_, privateCurrentSpeed());
    if (maybeTimeUntilExit) {
      // TODO: Need some way to reference the geometry from the event
      exitGeometryEventId_ = eventBroker_->publishDelayedEvent<event::EntityExitedGeometry>(std::chrono::milliseconds(static_cast<uint64_t>((*maybeTimeUntilExit)*1000)), globalId);
    }
  }
}

void MobileEntity::cancelGeometryEvents() {
  if (enterGeometryEventId_) {
    if (eventBroker_) {
      eventBroker_->cancelDelayedEvent(*enterGeometryEventId_);
    } else {
      LOG(WARNING) << "Trying to cancel enter geometry event, but do not have event broker";
    }
    enterGeometryEventId_.reset();
  }
  if (exitGeometryEventId_) {
    if (eventBroker_) {
      eventBroker_->cancelDelayedEvent(*exitGeometryEventId_);
    } else {
      LOG(WARNING) << "Trying to cancel exit geometry event, but do not have event broker";
    }
    exitGeometryEventId_.reset();
  }
}

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