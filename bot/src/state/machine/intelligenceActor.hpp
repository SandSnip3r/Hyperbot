#ifndef STATE_MACHINE_INTELLIGENCE_ACTOR_HPP_
#define STATE_MACHINE_INTELLIGENCE_ACTOR_HPP_

#include "broker/eventBroker.hpp"
#include "event/event.hpp"
#include "state/machine/stateMachine.hpp"

#include <silkroad_lib/pk2/ref/item.hpp>

#include <optional>
#include <string>

namespace state::machine {

class IntelligenceActor : public StateMachine {
public:
  IntelligenceActor(Bot &bot, sro::scalar_types::EntityGlobalId opponentGlobalId);
  ~IntelligenceActor() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"IntelligenceActor"};
  sro::scalar_types::EntityGlobalId opponentGlobalId_;
  std::optional<broker::EventBroker::EventId> requestTimeoutEventId_;

  void useItem(sro::pk2::ref::ItemId refId);
};

} // namespace state::machine

#endif // STATE_MACHINE_INTELLIGENCE_ACTOR_HPP_
