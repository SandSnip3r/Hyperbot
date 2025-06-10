#include "sequentialStateMachines.hpp"

#include "bot.hpp"
#include <absl/debugging/internal/demangle.h>
#include <typeinfo>

namespace state::machine {

SequentialStateMachines::SequentialStateMachines(Bot &bot) : StateMachine(bot) {}
SequentialStateMachines::SequentialStateMachines(StateMachine *parent) : StateMachine(parent) {}
SequentialStateMachines::~SequentialStateMachines() {}

Status SequentialStateMachines::onUpdate(const event::Event *event) {
  std::unique_lock lock(mutex_);
  if (!stateMachines_.empty()) {
    Status status = stateMachines_.front()->onUpdate(event);
    while (!stateMachines_.empty() && status == Status::kDone) {
      CHAR_VLOG(1) << "State machine is done; " << stateMachines_.size() << " left";
      // Remove this one.
      stateMachines_.pop_front();
      // Call the next one, if there is one.
      if (!stateMachines_.empty()) {
        status = stateMachines_.front()->onUpdate(event);
      }
    }
  }
  if (stateMachines_.empty()) {
    CHAR_VLOG(1) << "Last state machine is done";
    return Status::kDone;
  }
  CHAR_VLOG(2) << stateMachines_.size() << " state machines left. Not done";
  return Status::kNotDone;
}

std::string SequentialStateMachines::activeStateMachineName() const {
  std::unique_lock lock(mutex_);
  if (!stateMachines_.empty()) {
    return stateMachines_.front()->activeStateMachineName();
  }
  return absl::debugging_internal::DemangleString(typeid(*this).name());
}

// TODO: When pushing a state machine, set ourself as its parent.
// void SequentialStateMachines::push(std::unique_ptr<StateMachine> &&stateMachine) {
//   std::unique_lock lock(mutex_);
//   stateMachines_.push_back(std::move(stateMachine));
// }

} // namespace state::machine
