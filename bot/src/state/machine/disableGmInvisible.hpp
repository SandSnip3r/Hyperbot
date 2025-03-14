#ifndef STATE_MACHINE_DISABLE_GM_INVISIBLE_HPP_
#define STATE_MACHINE_DISABLE_GM_INVISIBLE_HPP_

#include "event/event.hpp"
#include "stateMachine.hpp"

#include <string>

namespace state::machine {

class DisableGmInvisible : public StateMachine {
public:
  DisableGmInvisible(StateMachine *parent);
  ~DisableGmInvisible() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"DisableGmInvisible"};
  bool initialized_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_DISABLE_GM_INVISIBLE_HPP_
