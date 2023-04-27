#ifndef STATE_MACHINE_WALKING_HPP_
#define STATE_MACHINE_WALKING_HPP_

#include "stateMachine.hpp"
#include "broker/eventBroker.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"

#include <silkroad_lib/position.h>

#include <optional>
#include <vector>

namespace state::machine {

// TODO: This feels like a weird home for this function.
std::vector<packet::building::NetworkReadyPosition> calculatePathToDestination(const sro::Position &destinationPosition, const Bot &bot);

class Walking : public StateMachine {
public:
  Walking(Bot &bot, const sro::Position &destinationPosition);
  ~Walking() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"Walking"};
  // We set this to true once we request a movement. If this is false and we're moving, we know that the current movement is not one that we started. This lets us interrupt movements already in progress.
  bool tookAction_{false};
  std::vector<packet::building::NetworkReadyPosition> waypoints_;
  size_t currentWaypointIndex_{0};
  std::optional<broker::EventBroker::DelayedEventId> movementRequestTimeoutEventId_;
};

} // namespace state::machine

#endif // STATE_MACHINE_WALKING_HPP_