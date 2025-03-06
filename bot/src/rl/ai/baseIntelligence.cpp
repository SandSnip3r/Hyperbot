#include "rl/ai/baseIntelligence.hpp"
#include "rl/trainingManager.hpp"

namespace rl::ai {

Observation BaseIntelligence::buildObservation(const Bot &bot, const event::Event *event, sro::scalar_types::ReferenceObjectId opponentGlobalId) const {
  return {bot, event, opponentGlobalId};
}

void BaseIntelligence::reportEventObservationAndAction(common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId observerGlobalId, const event::Event *event, const Observation &observation, int actionIndex) {
  trainingManager_.reportEventObservationAndAction(pvpId, observerGlobalId, event, observation, actionIndex);
}

} // namespace rl::ai