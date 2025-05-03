#include "bot.hpp"
#include "rl/ai/deepLearningIntelligence.hpp"
#include "rl/trainingManager.hpp"

#include <tracy/Tracy.hpp>

#include <absl/log/log.h>
#include <absl/log/vlog_is_on.h>

namespace rl::ai {

void DeepLearningIntelligence::resetForNewEpisode() {
  lastObservations_.clear();
}

int DeepLearningIntelligence::selectAction(Bot &bot, const Observation &observation, bool canSendPacket) {
  ZoneScopedN("DeepLearningIntelligence::selectAction");

  // Update epsilon.
  const float epsilon = getEpsilon();
  trainingManager_.getJaxInterface().addScalar("Epsilon", epsilon, stepCount_);
  ++stepCount_;

  if (lastObservations_.size() >= trainingManager_.getObservationStackSize()) {
    // We have enough observations, we can remove the oldest one.
    lastObservations_.pop_front();
  }
  // Add the new observation to the stack.
  lastObservations_.push_back(observation);

  std::bernoulli_distribution randomActionDistribution(epsilon);
  int actionIndex;
  if (randomActionDistribution(randomEngine_)) {
    // Do a random action.
    actionIndex = RandomIntelligence::selectAction(bot, observation, canSendPacket);
  } else {
    if (canSendPacket) {
      // Release the world state mutex while we call into JAX
      bot.worldState().mutex.unlock();
      std::vector<Observation> observationStack;
      for (const auto &obs : lastObservations_) {
        observationStack.push_back(obs);
      }
      actionIndex = trainingManager_.getJaxInterface().selectAction(trainingManager_.getObservationStackSize(), observationStack, canSendPacket);
      bot.worldState().mutex.lock();
    } else {
      // We cannot send a packet, we'll entirely side-step JAX and immediately return the do-nothing action
      // TODO: If we ever have more than one non-packet action, we should still call into JAX to let the model decide which to do.
      actionIndex = 0;
    }
  }
  return actionIndex;
}

float DeepLearningIntelligence::getEpsilon() {
  // For now, epsilon will decay linearly with the number of steps.
  return std::min(kInitialEpsilon, std::max(kFinalEpsilon, kInitialEpsilon - static_cast<float>(stepCount_) / kEpsilonDecaySteps));
}


} // namespace rl::ai
