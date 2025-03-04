#include "bot.hpp"
#include "rl/actionBuilder.hpp"
#include "rl/ai/randomIntelligence.hpp"

namespace rl::ai {

std::unique_ptr<Action> RandomIntelligence::selectAction(Bot &bot, const event::Event *event, sro::scalar_types::EntityGlobalId opponentGlobalId) {
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

  Observation observation = buildObservation(bot, event, opponentGlobalId);
  reportObservationAndAction(bot.selfState()->globalId, observation, actionIndex);

  return ActionBuilder::buildAction(bot, event, opponentGlobalId, actionIndex);
}

} // namespace rl::ai