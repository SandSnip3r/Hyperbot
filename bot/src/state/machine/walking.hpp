#ifndef STATE_MACHINE_WALKING_HPP_
#define STATE_MACHINE_WALKING_HPP_

#include "stateMachine.hpp"
#include "broker/eventBroker.hpp"
#include "packet/building/commonBuilding.hpp"

#include <silkroad_lib/position.hpp>

#include <optional>
#include <vector>

namespace state::machine {

class Walking : public StateMachine {
public:
  // TODO: Create a move constructor for the waypoints.
  Walking(Bot &bot, const std::vector<packet::building::NetworkReadyPosition> &waypoints);
  ~Walking() override;
  Status onUpdate(const event::Event *event) override;
  private:
  static inline std::string kName{"Walking"};
  // We set this to true once we request a movement. If this is false and we're moving, we know that the current movement is not one that we started. This lets us interrupt movements already in progress.
  bool initialized_{false};
  bool tookAction_{false};
  std::vector<packet::building::NetworkReadyPosition> waypoints_;
  size_t currentWaypointIndex_{0};
  std::optional<broker::EventBroker::EventId> movementRequestTimeoutEventId_;
  bool done() const;
};

} // namespace state::machine

#endif // STATE_MACHINE_WALKING_HPP_