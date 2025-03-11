#include "bot.hpp"
#include "rl/actionBuilder.hpp"
#include "rl/ai/randomIntelligence.hpp"

namespace rl::ai {

std::unique_ptr<Action> RandomIntelligence::selectAction(Bot &bot, state::machine::StateMachine *parentStateMachine, const event::Event *event, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId, bool canSendPacket) {
  Observation observation = buildObservation(bot, event, opponentGlobalId);
  int actionIndex;
  if (canSendPacket) {
    // Choose a truly random action.
    std::uniform_int_distribution<int> actionDist(0, ActionBuilder::actionSpaceSize()-1);
    actionIndex = actionDist(randomEngine_);
  } else {
    actionIndex = 0;
  }
  reportEventObservationAndAction(pvpId, bot.selfState()->globalId, event, observation, actionIndex);
  return ActionBuilder::buildAction(parentStateMachine, event, opponentGlobalId, actionIndex);
}

} // namespace rl::ai