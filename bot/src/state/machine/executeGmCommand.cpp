#include "executeGmCommand.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

ExecuteGmCommand::ExecuteGmCommand(Bot &bot, packet::enums::OperatorCommand gmCommand, PacketContainer gmCommandPacket) : StateMachine(bot), gmCommand_(gmCommand), gmCommandPacket_(gmCommandPacket) {
  // stateMachineCreated(kName);
  VLOG(1) << characterNameForLog() << " " << "ExecuteGmCommand created";
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
    VLOG(1) << characterNameForLog() << " " << "Injecting GM command packet";
    bot_.packetBroker().injectPacket(gmCommandPacket_, PacketContainer::Direction::kClientToServer);
    waitingForResponse_ = true;
    eventId_ = bot_.eventBroker().publishDelayedEvent(std::chrono::milliseconds(kMillisecondsTimeout), event::EventCode::kTimeout);
    return Status::kNotDone;
  }

  if (event == nullptr) {
    // No event, nothing to do.
    return Status::kNotDone;
  }
  if (event->eventCode == event::EventCode::kOperatorRequestSuccess) {
    const auto &castedEvent = dynamic_cast<const event::OperatorRequestSuccess&>(*event);
    if (castedEvent.globalId == bot_.selfState()->globalId && castedEvent.operatorCommand == gmCommand_) {
      VLOG(1) << characterNameForLog() << " " << "Successfully executed GM command";
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
      VLOG(1) << characterNameForLog() << " " << "GM command timed out";
      waitingForResponse_ = false;
      eventId_.reset();
      return onUpdate(event);
    }
  }
    return Status::kNotDone;
}

} // namespace state::machine
