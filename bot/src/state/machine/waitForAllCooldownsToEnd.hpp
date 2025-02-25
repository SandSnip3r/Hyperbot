#ifndef STATE_MACHINE_WAIT_FOR_ALL_COOLDOWNS_TO_END_HPP_
#define STATE_MACHINE_WAIT_FOR_ALL_COOLDOWNS_TO_END_HPP_

#include "event/event.hpp"
#include "state/machine/stateMachine.hpp"

#include <string>

namespace state::machine {

class WaitForAllCooldownsToEnd : public StateMachine {
public:
  WaitForAllCooldownsToEnd(Bot &bot);
  ~WaitForAllCooldownsToEnd() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"WaitForAllCooldownsToEnd"};
};

} // namespace state::machine

#endif // STATE_MACHINE_WAIT_FOR_ALL_COOLDOWNS_TO_END_HPP_
