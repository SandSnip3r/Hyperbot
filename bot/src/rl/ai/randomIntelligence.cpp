#include "bot.hpp"
#include "rl/actionBuilder.hpp"
#include "rl/ai/randomIntelligence.hpp"

namespace rl::ai {

std::unique_ptr<Action> RandomIntelligence::selectAction(Bot &bot, state::machine::StateMachine *parentStateMachine, const event::Event *event, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId) {
  Observation observation = buildObservation(bot, event, opponentGlobalId);

  int actionIndex;
  // Start with a high probability to do nothing/sleep.
  std::bernoulli_distribution sleepDist(0.85);
  if (sleepDist(randomEngine_)) {
    actionIndex = 0;
  } else {
    // Not sleeping, choose a true random action.
    std::uniform_int_distribution<int> actionDist(0, ActionBuilder::actionSpaceSize()-1);
    actionIndex = actionDist(randomEngine_);
  }
  reportEventObservationAndAction(pvpId, bot.selfState()->globalId, event, observation, actionIndex);
  return ActionBuilder::buildAction(parentStateMachine, event, opponentGlobalId, actionIndex);
}

} // namespace rl::ai