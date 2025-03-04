#ifndef RL_AI_BASE_INTELLIGENCE_HPP_
#define RL_AI_BASE_INTELLIGENCE_HPP_

#include "rl/action.hpp"
#include "rl/observation.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <memory>

namespace event {
struct Event;
} // namespace event

class Bot;

namespace rl {

class TrainingManager;

namespace ai {

class BaseIntelligence {
public:
  BaseIntelligence(TrainingManager &trainingManager) : trainingManager_(trainingManager) {}
  virtual std::unique_ptr<Action> selectAction(Bot &bot, const event::Event *event, sro::scalar_types::EntityGlobalId opponentGlobalId) = 0;
protected:
  Observation buildObservation(const Bot &bot, const event::Event *event, sro::scalar_types::ReferenceObjectId opponentGlobalId) const;
  void reportObservationAndAction(sro::scalar_types::EntityGlobalId observerGlobalId, const Observation &observation, int actionIndex);
private:
  TrainingManager &trainingManager_;
};

} // namespace ai
} // namespace rl

#endif // RL_AI_BASE_INTELLIGENCE_HPP_