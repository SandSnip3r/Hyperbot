#include "enablePvpMode.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentFreePvpUpdateRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

EnablePvpMode::EnablePvpMode(Bot &bot) : StateMachine(bot) {
}

EnablePvpMode::~EnablePvpMode() {
}

Status EnablePvpMode::onUpdate(const event::Event *event) {
  if (state_ == State::kInit) {
    const auto setPvpModePacket = packet::building::ClientAgentFreePvpUpdateRequest::setMode(packet::enums::FreePvpMode::kYellow);
    bot_.packetBroker().injectPacket(setPvpModePacket, PacketContainer::Direction::kClientToServer);
    VLOG(1) << characterNameForLog() << ' ' << "Sending packet to enable pvp";
    state_ = State::kSentRequest;
    return Status::kNotDone;
  } else {
    if (event != nullptr) {
      if (const auto *equipCountdownStartEvent = dynamic_cast<const event::EquipCountdownStart*>(event)) {
        if (equipCountdownStartEvent->globalId == bot_.selfState()->globalId) {
          if (state_ != State::kSentRequest) {
            LOG(WARNING) << "Expected to be in state \"kSentRequest\"";
          }
          VLOG(1) << characterNameForLog() << ' ' << "Received countdown start event";
          state_ = State::kCountdownRunning;
          return Status::kNotDone;
        }
      } else if (const auto *freePvpUpdateSuccessEvent = dynamic_cast<const event::FreePvpUpdateSuccess*>(event)) {
        if (freePvpUpdateSuccessEvent->globalId == bot_.selfState()->globalId) {
          if (state_ != State::kCountdownRunning) {
            LOG(WARNING) << "Expected to be in state \"kCountdownRunning\"";
          }
          VLOG(1) << characterNameForLog() << ' ' << "Free pvp response success! Done.";
          return Status::kDone;
        }
      }
    }
  }
  return Status::kNotDone;
}

} // namespace state::machine
