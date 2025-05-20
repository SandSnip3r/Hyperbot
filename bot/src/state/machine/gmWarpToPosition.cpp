#include "gmWarpToPosition.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"

#include <silkroad_lib/position_math.hpp>

#include <absl/log/log.h>

namespace state::machine {

GmWarpToPosition::GmWarpToPosition(StateMachine *parent, const sro::Position &position) : StateMachine(parent), position_(position) {
}

GmWarpToPosition::~GmWarpToPosition() {
}

Status GmWarpToPosition::onUpdate(const event::Event *event) {
  if (bot_.selfState() == nullptr) {
    // Not spawned, event cannot matter for us.
    return Status::kNotDone;
  }
  if (sro::position_math::calculateDistance2d(bot_.selfState()->position(), position_) < 1000.0f) {
    CHAR_VLOG(1) << "Already close enough to our target position, not warping";
    return Status::kDone;
  }
  if (event != nullptr) {
    if (const auto *operatorRequestSuccess = dynamic_cast<const event::OperatorRequestSuccess*>(event); operatorRequestSuccess != nullptr) {
      if (operatorRequestSuccess->globalId == bot_.selfState()->globalId && operatorRequestSuccess->operatorCommand == packet::enums::OperatorCommand::kWarpPoint) {
        CHAR_VLOG(2) << "Warp command successful";
        // Wait for the character to spawn.
      }
    } else if (const auto *operatorCommandError = dynamic_cast<const event::OperatorRequestError*>(event); operatorCommandError != nullptr) {
      if (operatorCommandError->globalId == bot_.selfState()->globalId && operatorCommandError->operatorCommand == packet::enums::OperatorCommand::kWarpPoint) {
        CHAR_VLOG(2) << "Warp command failed";
        bot_.eventBroker().cancelDelayedEvent(*eventId_);
        eventId_.reset();
        selfSpawned_ = false;
      }
    } else if (event->eventCode == event::EventCode::kTimeout) {
      CHAR_VLOG(2) << "Timed out waiting for warp command to complete";
      eventId_.reset();
      selfSpawned_ = false;
    } else if (const auto *selfSpawned = dynamic_cast<const event::SelfSpawned*>(event); selfSpawned != nullptr) {
      if (selfSpawned->sessionId == bot_.sessionId()) {
        CHAR_VLOG(2) << "Self spawned";
        selfSpawned_ = true;
        // Wait for the body state to change.
      }
    } else if (const auto *bodyStateChanged = dynamic_cast<const event::EntityBodyStateChanged*>(event); bodyStateChanged != nullptr) {
      if (selfSpawned_ && bodyStateChanged->globalId == bot_.selfState()->globalId) {
        CHAR_VLOG(1) << "Spawned & body state changed. Done";
        return Status::kDone;
      }
    }
  }
  if (eventId_) {
    // Waiting for warp to complete.
    return Status::kNotDone;
  }
  CHAR_VLOG(1) << "Warping to " << position_.toString();
  injectPacket(packet::building::ClientAgentOperatorRequest::warpPoint(position_, /*worldId=*/1), PacketContainer::Direction::kBotToServer);
  eventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(1500));
  return Status::kNotDone;
}

} // namespace state::machine
