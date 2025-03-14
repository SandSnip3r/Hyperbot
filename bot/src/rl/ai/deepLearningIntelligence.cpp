#include "bot.hpp"
#include "rl/ai/deepLearningIntelligence.hpp"
#include "rl/trainingManager.hpp"

#include <tracy/Tracy.hpp>

#include <absl/log/log.h>
#include <absl/log/vlog_is_on.h>

namespace rl::ai {

int DeepLearningIntelligence::selectAction(Bot &bot, const Observation &observation, bool canSendPacket) {
  ZoneScopedN("DeepLearningIntelligence::selectAction");
  const float epsilon = getEpsilon();
  if (VLOG_IS_ON(1) && stepCount_ <= kEpsilonDecaySteps) {
    std::bernoulli_distribution printDist(0.001);
    if (printDist(randomEngine_)) {
      VLOG(1) << "Step: " << stepCount_ << ", epsilon: " << epsilon;
    }
  }
  ++stepCount_;
  std::bernoulli_distribution randomActionDistribution(epsilon);
  if (randomActionDistribution(randomEngine_)) {
    // Do a random action.
    return RandomIntelligence::selectAction(bot, observation, canSendPacket);
  }
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

float DeepLearningIntelligence::getEpsilon() {
  // For now, epsilon will decay linearly with the number of steps.
  return std::max(kFinalEpsilon, kInitialEpsilon - static_cast<float>(stepCount_) / kEpsilonDecaySteps);
}


} // namespace rl::ai
