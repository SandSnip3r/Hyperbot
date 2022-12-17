#ifndef STATE_MACHINE_WALKING_HPP_
#define STATE_MACHINE_WALKING_HPP_

#include "stateMachine.hpp"
#include "broker/eventBroker.hpp"

#include <silkroad_lib/position.h>

#include <optional>
#include <vector>

namespace state::machine {

class Walking : public StateMachine {
public:
  Walking(Bot &bot, const sro::Position &destinationPosition);
  ~Walking() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"Walking"};
  std::vector<sro::Position> waypoints_;
  size_t currentWaypointIndex_{0};
  std::optional<broker::EventBroker::DelayedEventId> movementRequestTimeoutEventId_;
  std::vector<sro::Position> calculatePathToDestination(const sro::Position &destinationPosition) const;
};

} // namespace state::machine

#endif // STATE_MACHINE_WALKING_HPP_