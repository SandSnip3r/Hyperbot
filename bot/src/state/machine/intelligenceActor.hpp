#ifndef STATE_MACHINE_INTELLIGENCE_ACTOR_HPP_
#define STATE_MACHINE_INTELLIGENCE_ACTOR_HPP_

#include "broker/eventBroker.hpp"
#include "common/pvpDescriptor.hpp"
#include "event/event.hpp"
#include "rl/ai/baseIntelligence.hpp"
#include "state/machine/stateMachine.hpp"

#include <silkroad_lib/pk2/ref/item.hpp>

#include <optional>
#include <string>

namespace state::machine {

class IntelligenceActor : public StateMachine {
public:
  IntelligenceActor(Bot &bot, rl::ai::BaseIntelligence *intelligence, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId);
  ~IntelligenceActor() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"IntelligenceActor"};
  rl::ai::BaseIntelligence *intelligence_;
  const common::PvpDescriptor::PvpId pvpId_;
  const sro::scalar_types::EntityGlobalId opponentGlobalId_;
  std::optional<broker::EventBroker::EventId> sleepEventId_;

  void useItem(sro::pk2::ref::ItemId refId);
};

} // namespace state::machine

#endif // STATE_MACHINE_INTELLIGENCE_ACTOR_HPP_
