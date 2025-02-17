#include "executeGmCommand.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

ExecuteGmCommand::ExecuteGmCommand(Bot &bot, packet::enums::OperatorCommand gmCommand, PacketContainer gmCommandPacket) : StateMachine(bot), gmCommand_(gmCommand), gmCommandPacket_(gmCommandPacket) {
  // stateMachineCreated(kName);
  LOG(INFO) << "ExecuteGmCommand created";
}

ExecuteGmCommand::~ExecuteGmCommand() {
  // stateMachineDestroyed();
}

Status ExecuteGmCommand::onUpdate(const event::Event *event) {
  if (!waitingForResponse_) {
    LOG(INFO) << "Injecting GM command packet";
    bot_.packetBroker().injectPacket(gmCommandPacket_, PacketContainer::Direction::kClientToServer);
    waitingForResponse_ = true;
    return Status::kNotDone;
  }

  if (event == nullptr) {
    // No event, nothing to do.
    return Status::kNotDone;
  }
  if (event->eventCode == event::EventCode::kOperatorRequestSuccess) {
    const auto &castedEvent = dynamic_cast<const event::OperatorRequestSuccess&>(*event);
    if (castedEvent.operatorCommand == gmCommand_) {
      LOG(INFO) << "Successfully executed GM command";
      return Status::kDone;
    }
  } else if (event->eventCode == event::EventCode::kOperatorRequestError) {
    const auto &castedEvent = dynamic_cast<const event::OperatorRequestError&>(*event);
    if (castedEvent.operatorCommand == gmCommand_) {
      throw std::runtime_error("Failed to execute GM command!");
    }
  }
  return Status::kNotDone;
}

} // namespace state::machine
