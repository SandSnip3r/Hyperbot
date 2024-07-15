#ifndef STATE_MACHINE_CONCURRENT_STATE_MACHINES_HPP_
#define STATE_MACHINE_CONCURRENT_STATE_MACHINES_HPP_

#include "stateMachine.hpp"

#include <memory>
#include <vector>

namespace state::machine {

// Represents a set of state machines which are run concurrently over their lifetime. State machines' onUpdate()s are executed in an unspecified order.
class ConcurrentStateMachines : public StateMachine {
public:
  using StateMachine::StateMachine;
  ~ConcurrentStateMachines() override = default;
  void onUpdate(const event::Event *event) override;
  bool done() const override;

  template<typename StateMachineType, typename... Args>
  void emplace(Args&&... args) {
    stateMachines_.emplace_back(std::unique_ptr<StateMachineType>(new StateMachineType(bot_, std::forward<Args>(args)...)));
  }
private:
  std::vector<std::unique_ptr<StateMachine>> stateMachines_;
};

} // namespace state::machine

#endif // STATE_MACHINE_CONCURRENT_STATE_MACHINES_HPP_