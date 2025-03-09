#include "bot.hpp"
#include "rl/actionBuilder.hpp"
#include "rl/ai/deepLearningIntelligence.hpp"
#include "rl/trainingManager.hpp"

#include <tracy/Tracy.hpp>

namespace rl::ai {

std::unique_ptr<Action> DeepLearningIntelligence::selectAction(Bot &bot, state::machine::StateMachine *parentStateMachine, const event::Event *event, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId) {
  ZoneScopedN("DeepLearningIntelligence::selectAction");
  Observation observation = buildObservation(bot, event, opponentGlobalId);
  // Release the world state mutex while we call into JAX
  bot.worldState().mutex.unlock();
  int actionIndex = trainingManager_.getJaxInterface().selectAction(observation);
  bot.worldState().mutex.lock();
  reportEventObservationAndAction(pvpId, bot.selfState()->globalId, event, observation, actionIndex);
  return ActionBuilder::buildAction(parentStateMachine, event, opponentGlobalId, actionIndex);
}

} // namespace rl::ai
