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
  trainingManager_.getJaxInterface().addScalar("anneal/Epsilon", epsilon, stepCount_);
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

      // Create a ModelInput to pass to the JaxInterface
      ModelInput modelInput;
      modelInput.currentObservation = &observation;
      modelInput.pastObservationStack.reserve(lastObservations_.size() - 1);

      // Copy all but the current observation (which is the last one in lastObservations_)
      for (size_t i = 0; i < lastObservations_.size() - 1; ++i) {
        modelInput.pastObservationStack.push_back(&lastObservations_[i]);
      }

      actionIndex = trainingManager_.getJaxInterface().selectAction(modelInput, canSendPacket);
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
