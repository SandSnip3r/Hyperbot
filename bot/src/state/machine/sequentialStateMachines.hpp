#ifndef STATE_MACHINE_SEQUENTIAL_STATE_MACHINES_HPP_
#define STATE_MACHINE_SEQUENTIAL_STATE_MACHINES_HPP_

#include "event/event.hpp"
#include "stateMachine.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <deque>

namespace state::machine {

class SequentialStateMachines : public StateMachine {
public:
  SequentialStateMachines(Bot &bot);
  SequentialStateMachines(StateMachine *parent);
  ~SequentialStateMachines() override;
  Status onUpdate(const event::Event *event) override;
  std::string activeStateMachineName() const override;
  // void push(std::unique_ptr<StateMachine> &&stateMachine);

  template<typename StateMachineType, typename... Args>
  void emplace(Args&&... args) {
    std::unique_lock lock(mutex_);
    stateMachines_.emplace_back(std::unique_ptr<StateMachineType>(new StateMachineType(this, std::forward<Args>(args)...)));
  }
private:
  std::recursive_mutex mutex_;
  std::deque<std::unique_ptr<StateMachine>> stateMachines_;
};

} // namespace state::machine

#endif // STATE_MACHINE_SEQUENTIAL_STATE_MACHINES_HPP_
