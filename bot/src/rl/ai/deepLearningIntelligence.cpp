#include "bot.hpp"
#include "rl/actionBuilder.hpp"
#include "rl/ai/deepLearningIntelligence.hpp"
#include "rl/trainingManager.hpp"

namespace rl::ai {

std::unique_ptr<Action> DeepLearningIntelligence::selectAction(Bot &bot, const event::Event *event, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId) {
  Observation observation = buildObservation(bot, event, opponentGlobalId);
  int actionIndex = trainingManager_.getJaxInterface().selectAction(observation);
  reportEventObservationAndAction(pvpId, bot.selfState()->globalId, event, observation, actionIndex);
  return ActionBuilder::buildAction(bot, event, opponentGlobalId, actionIndex);
}

} // namespace rl::ai
