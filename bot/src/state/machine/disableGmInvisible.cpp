#include "disableGmInvisible.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"
#include "state/machine/executeGmCommand.hpp"

#include <absl/log/log.h>

namespace state::machine {

DisableGmInvisible::DisableGmInvisible(Bot &bot) : StateMachine(bot) {
}

DisableGmInvisible::~DisableGmInvisible() {
}

Status DisableGmInvisible::onUpdate(const event::Event *event) {
  // At first, if we're invisible, construct a child state machine to disable it.
  if (!initialized_) {
    initialized_ = true;
    if (bot_.selfState()->bodyState() == packet::enums::BodyState::kInvisibleGm) {
      VLOG(1) << characterNameForLog() << " " << "Constructing child state machine";
      // We just became invisible, toggle it off.
      setChildStateMachine<ExecuteGmCommand>(packet::enums::OperatorCommand::kInvisible, packet::building::ClientAgentOperatorRequest::toggleInvisible());
      return onUpdate(event);
    }
  }

  if (event != nullptr) {
    if (event->eventCode == event::EventCode::kEntityBodyStateChanged) {
      const auto *castedEvent = dynamic_cast<const event::EntityBodyStateChanged*>(event);
      if (castedEvent->globalId == bot_.selfState()->globalId) {
        if (bot_.selfState()->bodyState() == packet::enums::BodyState::kInvisibleGm) {
          VLOG(1) << characterNameForLog() << " " << "Just received event that we've become invisible";
          if (childState_ != nullptr) {
            VLOG(1) << characterNameForLog() << " " << "We're already in the process of toggling invisibility, skipping";
          } else {
            VLOG(1) << characterNameForLog() << " " << "Constructing child state machine";
            // We just became invisible, toggle it off.
            setChildStateMachine<ExecuteGmCommand>(packet::enums::OperatorCommand::kInvisible, packet::building::ClientAgentOperatorRequest::toggleInvisible());
            return onUpdate(event);
          }
        } else {
          VLOG(1) << characterNameForLog() << " " << "Body state updated and we're not invisible. Have child? " << (childState_ != nullptr);
          return Status::kDone;
        }
      }
    }
  }

  if (childState_ != nullptr) {
    VLOG(1) << characterNameForLog() << " " << "Running child state machine";
    const Status status = childState_->onUpdate(event);
    if (status == Status::kDone) {
      childState_.reset();
      VLOG(1) << characterNameForLog() << " " << "  GM command completed";
    }
    return Status::kNotDone;
  }

  return Status::kNotDone;
}

} // namespace state::machine
