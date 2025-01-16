#include "concurrentStateMachines.hpp"

namespace state::machine {

void ConcurrentStateMachines::onUpdate(const event::Event *event) {
  for (size_t i=0; i<stateMachines_.size();) {
    stateMachines_.at(i)->onUpdate(event);
    if (stateMachines_.at(i)->done()) {
      // Remove from list
      stateMachines_.erase(stateMachines_.begin()+i);
    } else {
      ++i;
    }
  }
}

bool ConcurrentStateMachines::done() const {
  return stateMachines_.empty();
}

} // namespace state::machine