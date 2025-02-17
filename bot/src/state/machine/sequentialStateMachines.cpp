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
    const Status status = stateMachines_.front()->onUpdate(event);
    if (status == Status::kDone) {
      stateMachines_.pop_front();
      return onUpdate(event);
    }
  }
  if (stateMachines_.empty()) {
    return Status::kDone;
  } else {
    return Status::kNotDone;
  }
}

} // namespace state::machine
