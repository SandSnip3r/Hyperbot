#include "sequentialStateMachines.hpp"

#include "bot.hpp"

namespace state::machine {

SequentialStateMachines::SequentialStateMachines(Bot &bot) : StateMachine(bot) {
}

SequentialStateMachines::~SequentialStateMachines() {
}

Status SequentialStateMachines::onUpdate(const event::Event *event) {
  std::unique_lock lock(mutex_);
  if (!stateMachines_.empty()) {
    Status status = stateMachines_.front()->onUpdate(event);
    while (!stateMachines_.empty() && status == Status::kDone) {
      VLOG(1) << "State machine is done; " << stateMachines_.size() << " left";
      // Remove this one.
      stateMachines_.pop_front();
      // Call the next one, if there is one.
      if (!stateMachines_.empty()) {
        status = stateMachines_.front()->onUpdate(event);
      }
    }
  }
  if (stateMachines_.empty()) {
    return Status::kDone;
  } else {
    return Status::kNotDone;
  }
}

void SequentialStateMachines::push(std::unique_ptr<StateMachine> &&stateMachine) {
  std::unique_lock lock(mutex_);
  stateMachines_.push_back(std::move(stateMachine));
}

} // namespace state::machine
