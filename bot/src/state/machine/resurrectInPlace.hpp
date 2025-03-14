#ifndef STATE_MACHINE_RESURRECT_IN_PLACE_HPP_
#define STATE_MACHINE_RESURRECT_IN_PLACE_HPP_

#include "broker/eventBroker.hpp"
#include "event/event.hpp"
#include "state/machine/stateMachine.hpp"

#include <optional>
#include <string>

namespace state::machine {

class ResurrectInPlace : public StateMachine {
public:
  ResurrectInPlace(StateMachine *parent, bool receivedResurrectionOptionAlready);
  ~ResurrectInPlace() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"ResurrectInPlace"};
  const bool receivedResurrectionOptionAlready_;
  std::optional<broker::EventBroker::EventId> requestTimeoutEventId_;
  void sendResurrectRequest();
};

} // namespace state::machine

#endif // STATE_MACHINE_RESURRECT_IN_PLACE_HPP_
