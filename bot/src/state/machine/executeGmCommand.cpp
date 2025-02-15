#include "executeGmCommand.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

ExecuteGmCommand ExecuteGmCommand::makeItem(Bot &bot, sro::scalar_types::ReferenceObjectId refItemId, uint8_t optLevelOrAmount) {
  return ExecuteGmCommand(bot, packet::enums::OperatorCommand::kMakeItem, packet::building::ClientAgentOperatorRequest::makeItem(refItemId, optLevelOrAmount));
}

ExecuteGmCommand::ExecuteGmCommand(Bot &bot, packet::enums::OperatorCommand gmCommand, PacketContainer gmCommandPacket) : StateMachine(bot), gmCommand_(gmCommand), gmCommandPacket_(gmCommandPacket) {
  stateMachineCreated(kName);
  // const PacketContainer gmCommandPacket_;
  bot_.packetBroker().injectPacket(gmCommandPacket_, PacketContainer::Direction::kClientToServer);
}

ExecuteGmCommand::~ExecuteGmCommand() {
  stateMachineDestroyed();
}

void ExecuteGmCommand::onUpdate(const event::Event *event) {
  if (event == nullptr) {
    // No event, nothing to do.
    return;
  }
  if (event->eventCode == event::EventCode::kOperatorRequestSuccess) {
    const auto &castedEvent = dynamic_cast<const event::OperatorRequestSuccess&>(*event);
    if (castedEvent.operatorCommand == gmCommand_) {
      LOG(INFO) << "Successfully executed GM command";
      done_ = true;
    }
  } else if (event->eventCode == event::EventCode::kOperatorRequestError) {
    const auto &castedEvent = dynamic_cast<const event::OperatorRequestError&>(*event);
    if (castedEvent.operatorCommand == gmCommand_) {
      throw std::runtime_error("Failed to execute GM command!");
    }
  }
}

bool ExecuteGmCommand::done() const {
  return done_;
}

} // namespace state::machine
