#include "executeGmCommand.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

ExecuteGmCommand::ExecuteGmCommand(StateMachine *parent, packet::enums::OperatorCommand gmCommand, PacketContainer gmCommandPacket) : StateMachine(parent), gmCommand_(gmCommand), gmCommandPacket_(gmCommandPacket) {
  CHAR_VLOG(1) << "ExecuteGmCommand created";
}

ExecuteGmCommand::~ExecuteGmCommand() {
  if (eventId_) {
    bot_.eventBroker().cancelDelayedEvent(*eventId_);
  }
}

Status ExecuteGmCommand::onUpdate(const event::Event *event) {
  constexpr int kMillisecondsTimeout{1000};
  if (!waitingForResponse_) {
    CHAR_VLOG(1) << "Injecting GM command packet";
    injectPacket(gmCommandPacket_, PacketContainer::Direction::kBotToServer);
    waitingForResponse_ = true;
    eventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(kMillisecondsTimeout));
    return Status::kNotDone;
  }

  if (event == nullptr) {
    // No event, nothing to do.
    return Status::kNotDone;
  }
  if (const event::OperatorRequestSuccess *operatorRequestSuccessEvent = dynamic_cast<const event::OperatorRequestSuccess*>(event); operatorRequestSuccessEvent != nullptr) {
    if (operatorRequestSuccessEvent->globalId == bot_.selfState()->globalId && operatorRequestSuccessEvent->operatorCommand == gmCommand_) {
      CHAR_VLOG(1) << "Successfully executed GM command";
      return Status::kDone;
    }
  } else if (const event::OperatorRequestError *operatorRequestErrorEvent = dynamic_cast<const event::OperatorRequestError*>(event); operatorRequestErrorEvent != nullptr) {
    if (operatorRequestErrorEvent->globalId == bot_.selfState()->globalId && operatorRequestErrorEvent->operatorCommand == gmCommand_) {
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
