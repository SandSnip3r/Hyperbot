#include "bot.hpp"
#include "rl/actionBuilder.hpp"
#include "rl/ai/randomIntelligence.hpp"

namespace rl::ai {

int RandomIntelligence::selectAction(Bot &bot, const Observation &observation, bool canSendPacket) {
  int actionIndex;
  if (canSendPacket) {
    // Choose a truly random action.
    std::uniform_int_distribution<int> actionDist(0, ActionBuilder::actionSpaceSize()-1);
    actionIndex = actionDist(randomEngine_);
  } else {
    actionIndex = 0;
  }
  return actionIndex;
}

} // namespace rl::ai