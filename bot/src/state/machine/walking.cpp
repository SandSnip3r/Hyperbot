#include "walking.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"

#include <silkroad_lib/position_math.hpp>

#include <absl/log/log.h>

namespace state::machine {

Walking::Walking(StateMachine *parent, const std::vector<packet::building::NetworkReadyPosition> &waypoints) : StateMachine(parent), waypoints_(waypoints) {}

Walking::~Walking() {
  if (movementRequestTimeoutEventId_) {
    bot_.eventBroker().cancelDelayedEvent(*movementRequestTimeoutEventId_);
    movementRequestTimeoutEventId_.reset();
  }
  bot_.eventBroker().publishEvent<event::WalkingPathUpdated>(std::vector<packet::building::NetworkReadyPosition>());
}

Status Walking::onUpdate(const event::Event *event) {
  if (!initialized_) {
    initialized_ = true;
    bot_.eventBroker().publishEvent<event::WalkingPathUpdated>(std::vector<packet::building::NetworkReadyPosition>(waypoints_.begin(), waypoints_.end()));
    pushBlockedOpcode(packet::Opcode::kClientAgentCharacterMoveRequest);
  }
  if (event != nullptr) {
    if (const auto *movementBeganEvent = dynamic_cast<const event::EntityMovementBegan*>(event); movementBeganEvent != nullptr && movementBeganEvent->globalId == bot_.selfState()->globalId) {
      // We started to move, our movement request must've been successful
      if (movementRequestTimeoutEventId_) {
        CHAR_VLOG(1) << "Movement began, cancelling movement request timeout timer";
        bot_.eventBroker().cancelDelayedEvent(*movementRequestTimeoutEventId_);
        movementRequestTimeoutEventId_.reset();
      } else {
        // TODO: This triggers because two different bots receive the movement began packet and both try to update the mobile entity.
        CHAR_VLOG(1) << "Movement began, but had no running movement request timeout timer";
      }
      // Nothing else to do here. We're now waiting for our movement to end
      return Status::kNotDone;
    } else if (const auto *movementEndedEvent = dynamic_cast<const event::EntityMovementEnded*>(event); movementEndedEvent != nullptr && movementEndedEvent->globalId == bot_.selfState()->globalId) {
      // If we send a request to move, but get knocked back before the MovementBegin happens, the knockback movement will send this MovementEnded event
      if (movementRequestTimeoutEventId_) {
        CHAR_VLOG(1) << "Movement ended, cancelling delayed event";
        bot_.eventBroker().cancelDelayedEvent(*movementRequestTimeoutEventId_);
        movementRequestTimeoutEventId_.reset();
      }
    } else if (event->eventCode == event::EventCode::kMovementRequestTimedOut) {
      CHAR_VLOG(1) << "Movement request timed out";
      movementRequestTimeoutEventId_.reset();
    }
  }

  if (tookAction_ && bot_.selfState()->moving()) {
    // Still moving, nothing to do
    return Status::kNotDone;
  }

  // We're not moving
  // Did we just arrive at this waypoint?
  bool updatedCurrentWaypoint{false};
  while (currentWaypointIndex_ < waypoints_.size() && sro::position_math::calculateDistance2d(bot_.selfState()->position(), waypoints_.at(currentWaypointIndex_).asSroPosition()) < sqrt(0.5)) {
    // Already at this waypoint, increment index
    ++currentWaypointIndex_;
    updatedCurrentWaypoint = true;
  }
  if (done()) {
    // Finished walking
    return Status::kDone;
  }
  if (updatedCurrentWaypoint) {
    bot_.eventBroker().publishEvent<event::WalkingPathUpdated>(std::vector<packet::building::NetworkReadyPosition>(waypoints_.begin()+currentWaypointIndex_-1, waypoints_.end()));
  }

  // Not yet done walking
  if (movementRequestTimeoutEventId_) {
    // Already asked to move, nothing to do
    return Status::kNotDone;
  }

  if (!canMove()) {
    return Status::kNotDone;
  }

  // We are not moving, we're not at the current waypoint, and there's not a pending movement request
  // Send a request to move to the current waypoint
  const auto &currentWaypoint = waypoints_.at(currentWaypointIndex_);
  CHAR_VLOG(1) << "Requesting movement to " << currentWaypoint.asSroPosition() << ". We are currently at " << bot_.selfState()->position() << " which is " << sro::position_math::calculateDistance2d(currentWaypoint.asSroPosition(), bot_.selfState()->position()) << 'm';
  const auto movementPacket = packet::building::ClientAgentCharacterMoveRequest::moveToPosition(currentWaypoint);
  injectPacket(movementPacket, PacketContainer::Direction::kBotToServer);
  const int kMovementRequestTimeoutMs{333}; // TODO: Move somewhere else and make an educated guess about what this value should be
  movementRequestTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kMovementRequestTimedOut, std::chrono::milliseconds(kMovementRequestTimeoutMs));
  tookAction_ = true;
  return Status::kNotDone;
}

bool Walking::done() const {
  return (currentWaypointIndex_ == waypoints_.size());
}

} // namespace state::machine