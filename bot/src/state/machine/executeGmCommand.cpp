#include "executeGmCommand.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

ExecuteGmCommand::ExecuteGmCommand(Bot &bot, packet::enums::OperatorCommand gmCommand, PacketContainer gmCommandPacket) : StateMachine(bot), gmCommand_(gmCommand), gmCommandPacket_(gmCommandPacket) {
  // stateMachineCreated(kName);
  CHAR_VLOG(1) << "ExecuteGmCommand created";
}

ExecuteGmCommand::~ExecuteGmCommand() {
  // stateMachineDestroyed();
  if (eventId_) {
    bot_.eventBroker().cancelDelayedEvent(*eventId_);
  }
}

Status ExecuteGmCommand::onUpdate(const event::Event *event) {
  constexpr int kMillisecondsTimeout{1000};
  if (!waitingForResponse_) {
    CHAR_VLOG(1) << "Injecting GM command packet";
    bot_.packetBroker().injectPacket(gmCommandPacket_, PacketContainer::Direction::kBotToServer);
    waitingForResponse_ = true;
    eventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(kMillisecondsTimeout));
    return Status::kNotDone;
  }

  if (event == nullptr) {
    // No event, nothing to do.
    return Status::kNotDone;
  }
  if (event->eventCode == event::EventCode::kOperatorRequestSuccess) {
    const auto &castedEvent = dynamic_cast<const event::OperatorRequestSuccess&>(*event);
    if (castedEvent.globalId == bot_.selfState()->globalId && castedEvent.operatorCommand == gmCommand_) {
      CHAR_VLOG(1) << "Successfully executed GM command";
      return Status::kDone;
    }
  } else if (event->eventCode == event::EventCode::kOperatorRequestError) {
    const auto &castedEvent = dynamic_cast<const event::OperatorRequestError&>(*event);
    if (castedEvent.globalId == bot_.selfState()->globalId && castedEvent.operatorCommand == gmCommand_) {
      throw std::runtime_error("Failed to execute GM command!");
    }
  } else if (event->eventCode == event::EventCode::kTimeout) {
    if (eventId_ && event->eventId == *eventId_) {
      // This is our timeout!
      CHAR_VLOG(1) << "GM command timed out";
      waitingForResponse_ = false;
      eventId_.reset();
      return onUpdate(event);
    }
  }
    return Status::kNotDone;
}

} // namespace state::machine
