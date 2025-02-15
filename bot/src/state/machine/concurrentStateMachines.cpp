#include "concurrentStateMachines.hpp"

namespace state::machine {

Status ConcurrentStateMachines::onUpdate(const event::Event *event) {
  for (size_t i=0; i<stateMachines_.size();) {
    Status status = stateMachines_.at(i)->onUpdate(event);
    if (status == Status::kDone) {
      // Remove from list
      stateMachines_.erase(stateMachines_.begin()+i);
    } else {
      ++i;
    }
  }
  if (stateMachines_.empty()) {
    return Status::kDone;
  } else {
    return Status::kNotDone;
  }
}

} // namespace state::machine