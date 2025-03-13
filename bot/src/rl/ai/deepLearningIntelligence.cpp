#include "bot.hpp"
#include "rl/ai/deepLearningIntelligence.hpp"
#include "rl/trainingManager.hpp"

#include <tracy/Tracy.hpp>

namespace rl::ai {

int DeepLearningIntelligence::selectAction(Bot &bot, const Observation &observation, bool canSendPacket) {
  ZoneScopedN("DeepLearningIntelligence::selectAction");
  int actionIndex;
  if (canSendPacket) {
    // Release the world state mutex while we call into JAX
    bot.worldState().mutex.unlock();
    actionIndex = trainingManager_.getJaxInterface().selectAction(observation, canSendPacket);
    bot.worldState().mutex.lock();
  } else {
    // We cannot send a packet, we'll entirely side-step JAX and immediately return the do-nothing action
    // TODO: If we ever have more than one non-packet action, we should still call into JAX to let the model decide which to do.
    actionIndex = 0;
  }
  return actionIndex;
}

} // namespace rl::ai
