#ifndef STATE_MACHINE_GM_WARP_TO_POSITION_HPP_
#define STATE_MACHINE_GM_WARP_TO_POSITION_HPP_

#include "event/event.hpp"
#include "state/machine/stateMachine.hpp"

#include <optional>
#include <string>

namespace state::machine {

class GmWarpToPosition : public StateMachine {
public:
  GmWarpToPosition(StateMachine *parent, const sro::Position &position);
  ~GmWarpToPosition() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"GmWarpToPosition"};
  const sro::Position position_;
  bool selfSpawned_{false};
  std::optional<broker::EventBroker::EventId> eventId_;
};

} // namespace state::machine

#endif // STATE_MACHINE_GM_WARP_TO_POSITION_HPP_
