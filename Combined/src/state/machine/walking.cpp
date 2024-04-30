#include "walking.hpp"

#include "bot.hpp"
#include "logging.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"

#include <silkroad_lib/position_math.h>

namespace state::machine {

Walking::Walking(Bot &bot, const std::vector<packet::building::NetworkReadyPosition> &waypoints) : StateMachine(bot), waypoints_(waypoints) {
  stateMachineCreated(kName);
  bot_.eventBroker().publishEvent<event::WalkingPathUpdated>(std::vector<packet::building::NetworkReadyPosition>(waypoints_.begin(), waypoints_.end()));
  pushBlockedOpcode(packet::Opcode::kClientAgentCharacterMoveRequest);
}

Walking::~Walking() {
  bot_.eventBroker().publishEvent<event::WalkingPathUpdated>(std::vector<packet::building::NetworkReadyPosition>());
  stateMachineDestroyed();
}

void Walking::onUpdate(const event::Event *event) {
  if (done()) {
    HYPERBOT_LOG() << "Walking but done" << std::endl;
    return;
  }

  if (event != nullptr) {
    if (const auto *movementBeganEvent = dynamic_cast<const event::EntityMovementBegan*>(event); movementBeganEvent != nullptr && movementBeganEvent->globalId == bot_.selfState().globalId) {
      // We started to move, our movement request must've been successful
      if (movementRequestTimeoutEventId_) {
        bot_.eventBroker().cancelDelayedEvent(*movementRequestTimeoutEventId_);
        movementRequestTimeoutEventId_.reset();
      } else {
        HYPERBOT_LOG() << "Movement began, but had no running movement request timeout timer" << std::endl;
      }
      // Nothing else to do here. We're now waiting for our movement to end
      return;
    } else if (const auto *movementEndedEvent = dynamic_cast<const event::EntityMovementEnded*>(event); movementEndedEvent != nullptr && movementEndedEvent->globalId == bot_.selfState().globalId) {
      // If we send a request to move, but get knocked back before the MovementBegin happens, the knockback movement will send this MovementEnded event
      if (movementRequestTimeoutEventId_) {
        bot_.eventBroker().cancelDelayedEvent(*movementRequestTimeoutEventId_);
        movementRequestTimeoutEventId_.reset();
      }
    } else if (event->eventCode == event::EventCode::kMovementRequestTimedOut) {
      HYPERBOT_LOG() << "kMovementRequestTimedOut" << std::endl;
      HYPERBOT_LOG() << "Movement request timed out" << std::endl;
      movementRequestTimeoutEventId_.reset();
    }
  }

  if (tookAction_ && bot_.selfState().moving()) {
    // Still moving, nothing to do
    return;
  }

  // We're not moving
  // Did we just arrive at this waypoint?
  bool updatedCurrentWaypoint{false};
  while (currentWaypointIndex_ < waypoints_.size() && sro::position_math::calculateDistance2d(bot_.selfState().position(), waypoints_.at(currentWaypointIndex_).asSroPosition()) < sqrt(0.5)) {
    // Already at this waypoint, increment index
    ++currentWaypointIndex_;
    updatedCurrentWaypoint = true;
  }
  if (done()) {
    // Finished walking
    return;
  }
  if (updatedCurrentWaypoint) {
    bot_.eventBroker().publishEvent<event::WalkingPathUpdated>(std::vector<packet::building::NetworkReadyPosition>(waypoints_.begin()+currentWaypointIndex_-1, waypoints_.end()));
  }

  // Not yet done walking
  if (movementRequestTimeoutEventId_) {
    // Already asked to move, nothing to do
    return;
  }

  if (!canMove()) {
    return;
  }

  // We are not moving, we're not at the current waypoint, and there's not a pending movement request
  // Send a request to move to the current waypoint
  const auto &currentWaypoint = waypoints_.at(currentWaypointIndex_);
  // HYPERBOT_LOG() << "Requesting movement to " << currentWaypoint.asSroPosition() << ". We are currently at " << bot_.selfState().position() << " which is " << sro::position_math::calculateDistance2d(currentWaypoint.asSroPosition(), bot_.selfState().position()) << 'm' << std::endl;
  const auto movementPacket = packet::building::ClientAgentCharacterMoveRequest::moveToPosition(currentWaypoint);
  bot_.packetBroker().injectPacket(movementPacket, PacketContainer::Direction::kClientToServer);
  const int kMovementRequestTimeoutMs{333}; // TODO: Move somewhere else and make an educated guess about what this value should be
  movementRequestTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent(std::chrono::milliseconds(kMovementRequestTimeoutMs), event::EventCode::kMovementRequestTimedOut);
  tookAction_ = true;
}

bool Walking::done() const {
  return (currentWaypointIndex_ == waypoints_.size());
}

} // namespace state::machine