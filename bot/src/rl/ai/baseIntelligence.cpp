#include "rl/ai/baseIntelligence.hpp"
#include "rl/trainingManager.hpp"

namespace rl::ai {

Observation BaseIntelligence::buildObservation(const Bot &bot, const event::Event *event, sro::scalar_types::ReferenceObjectId opponentGlobalId) const {
  return {bot, event, opponentGlobalId};
}

void BaseIntelligence::reportObservationAndAction(sro::scalar_types::EntityGlobalId observerGlobalId, const Observation &observation, int actionIndex) {
  trainingManager_.reportObservationAndAction(observerGlobalId, observation, actionIndex);
}

} // namespace rl::ai