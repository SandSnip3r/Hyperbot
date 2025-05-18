#include "disableGmInvisible.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"
#include "state/machine/executeGmCommand.hpp"

#include <absl/log/log.h>

namespace state::machine {

DisableGmInvisible::DisableGmInvisible(StateMachine *parent) : StateMachine(parent) {}
DisableGmInvisible::~DisableGmInvisible() {}

Status DisableGmInvisible::onUpdate(const event::Event *event) {
  // At first, if we're invisible, construct a child state machine to disable it.
  if (!initialized_) {
    initialized_ = true;
    if (bot_.selfState()->bodyState() != packet::enums::BodyState::kInvisibleGm) {
      CHAR_VLOG(1) << "Not invisible, done";
      return Status::kDone;
    }
    if (bot_.selfState()->bodyState() == packet::enums::BodyState::kInvisibleGm) {
      CHAR_VLOG(1) << "Constructing child state machine";
      // We just became invisible, toggle it off.
      setChildStateMachine<ExecuteGmCommand>(packet::enums::OperatorCommand::kInvisible, packet::building::ClientAgentOperatorRequest::toggleInvisible());
      return onUpdate(event);
    }
  }

  if (event != nullptr) {
    if (event->eventCode == event::EventCode::kEntityBodyStateChanged) {
      const event::EntityBodyStateChanged *castedEvent = dynamic_cast<const event::EntityBodyStateChanged*>(event);
      if (castedEvent->globalId == bot_.selfState()->globalId) {
        if (bot_.selfState()->bodyState() == packet::enums::BodyState::kInvisibleGm) {
          CHAR_VLOG(1) << "Just received event that we've become invisible";
          if (childState_ != nullptr) {
            CHAR_VLOG(1) << "We're already in the process of toggling invisibility, skipping";
          } else {
            CHAR_VLOG(1) << "Constructing child state machine";
            // We just became invisible, toggle it off.
            setChildStateMachine<ExecuteGmCommand>(packet::enums::OperatorCommand::kInvisible, packet::building::ClientAgentOperatorRequest::toggleInvisible());
            return onUpdate(event);
          }
        } else {
          CHAR_VLOG(1) << "Body state updated and we're not invisible. Have child? " << (childState_ != nullptr);
          return Status::kDone;
        }
      }
    }
  }

  if (childState_ != nullptr) {
    CHAR_VLOG(1) << "Running child state machine";
    const Status status = childState_->onUpdate(event);
    if (status == Status::kDone) {
      childState_.reset();
      CHAR_VLOG(1) << "  GM command completed";
    }
    return Status::kNotDone;
  }

  return Status::kNotDone;
}

} // namespace state::machine
