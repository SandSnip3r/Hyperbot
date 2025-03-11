#ifndef RL_AI_BASE_INTELLIGENCE_HPP_
#define RL_AI_BASE_INTELLIGENCE_HPP_

#include "common/pvpDescriptor.hpp"
#include "rl/action.hpp"
#include "rl/observation.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <memory>

namespace event {
struct Event;
} // namespace event

namespace state::machine {
class StateMachine;
} // namespace state::machine

class Bot;

namespace rl {

class TrainingManager;

namespace ai {

class BaseIntelligence {
public:
  BaseIntelligence(TrainingManager &trainingManager) : trainingManager_(trainingManager) {}
  virtual std::unique_ptr<Action> selectAction(Bot &bot, state::machine::StateMachine *parentStateMachine, const event::Event *event, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId, bool canSendPacket) = 0;
  virtual std::string_view name() const = 0;

protected:
  TrainingManager &trainingManager_;

  Observation buildObservation(const Bot &bot, const event::Event *event, sro::scalar_types::ReferenceObjectId opponentGlobalId) const;
  void reportEventObservationAndAction(common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId observerGlobalId, const event::Event *event, const Observation &observation, int actionIndex);
};

} // namespace ai
} // namespace rl

#endif // RL_AI_BASE_INTELLIGENCE_HPP_