#include "enablePvpMode.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentFreePvpUpdateRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

EnablePvpMode::EnablePvpMode(StateMachine *parent) : StateMachine(parent) {}

EnablePvpMode::~EnablePvpMode() {
  if (requestTimeoutEventId_) {
    bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
    requestTimeoutEventId_.reset();
  }
}

Status EnablePvpMode::onUpdate(const event::Event *event) {
  if (state_ == State::kInit) {
    if (bot_.selfState()->freePvpMode == packet::enums::FreePvpMode::kYellow) {
      CHAR_VLOG(1) << "Already in PVP mode";
      return Status::kDone;
    }
    sendRequest();
    return Status::kNotDone;
  } else {
    if (event != nullptr) {
      if (const event::EquipCountdownStart *equipCountdownStartEvent = dynamic_cast<const event::EquipCountdownStart*>(event); equipCountdownStartEvent != nullptr) {
        if (equipCountdownStartEvent->globalId == bot_.selfState()->globalId) {
          if (state_ != State::kSentRequest) {
            LOG(WARNING) << "Expected to be in state \"kSentRequest\"";
          }
          CHAR_VLOG(1) << "Received countdown start event";
          if (requestTimeoutEventId_) {
            bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
            requestTimeoutEventId_.reset();
          }
          state_ = State::kCountdownRunning;
          return Status::kNotDone;
        }
      } else if (const event::FreePvpUpdateSuccess *freePvpUpdateSuccessEvent = dynamic_cast<const event::FreePvpUpdateSuccess*>(event); freePvpUpdateSuccessEvent != nullptr) {
        if (freePvpUpdateSuccessEvent->globalId == bot_.selfState()->globalId) {
          if (state_ != State::kCountdownRunning) {
            LOG(WARNING) << "Expected to be in state \"kCountdownRunning\"";
          }
          CHAR_VLOG(1) << "Free pvp response success! Done.";
          return Status::kDone;
        }
      } else if (event->eventCode == event::EventCode::kTimeout &&
                 requestTimeoutEventId_ &&
                 *requestTimeoutEventId_ == event->eventId) {
        CHAR_VLOG(1) << "Free pvp response timeout. Trying again";
        sendRequest();
        return Status::kNotDone;
      }
    }
  }
  return Status::kNotDone;
}

void EnablePvpMode::sendRequest() {
  const auto setPvpModePacket = packet::building::ClientAgentFreePvpUpdateRequest::setMode(packet::enums::FreePvpMode::kYellow);
  injectPacket(setPvpModePacket, PacketContainer::Direction::kBotToServer);
  requestTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(666));
  CHAR_VLOG(1) << "Sending packet to enable pvp";
  state_ = State::kSentRequest;
}

} // namespace state::machine
