#include "bot.hpp"
#include "rl/ai/deepLearningIntelligence.hpp"
#include "rl/modelInputs.hpp"
#include "rl/trainingManager.hpp"

#include <tracy/Tracy.hpp>

#include <absl/log/log.h>
#include <absl/log/vlog_is_on.h>

namespace rl::ai {

int DeepLearningIntelligence::selectAction(Bot &bot, const Observation &observation, bool canSendPacket, std::optional<std::string> metadata) {
  ZoneScopedN("DeepLearningIntelligence::selectAction");

  // Update epsilon.
  const float epsilon = trainingManager_.getEpsilon();
  std::bernoulli_distribution randomActionDistribution(epsilon);
  int actionIndex;
  if (randomActionDistribution(randomEngine_)) {
    // Do a random action.
    actionIndex = RandomIntelligence::selectAction(bot, observation, canSendPacket);
    VLOG(1) << "Acting randomly (esp=" << epsilon << ") with action " << actionIndex;
  } else {
    if (canSendPacket) {
      // Release the world state mutex while we call into JAX
      bot.worldState().mutex.unlock();

      // Create a ModelInput to pass to the JaxInterface
      model_inputs::ModelInputView modelInputView;
      modelInputView.currentObservation = &observation;
      modelInputView.pastObservationStack.reserve(pastObservationsAndActions_.size());
      modelInputView.pastActionStack.reserve(pastObservationsAndActions_.size());

      // Fill the model input with the past observations and actions.
      for (size_t i=0; i<pastObservationsAndActions_.size(); ++i) {
        modelInputView.pastObservationStack.push_back(&pastObservationsAndActions_[i].first);
        modelInputView.pastActionStack.push_back(pastObservationsAndActions_[i].second);
      }

      rl::JaxInterface::ActionSelectionResult result = trainingManager_.getJaxInterface().selectAction(modelInputView, canSendPacket, metadata);
      actionIndex = result.actionIndex;
      VLOG(1) << "Can send packet, queried model and taking action " << actionIndex;
      bot.sendQValues(result.qValues);
      bot.worldState().mutex.lock();
    } else {
      // We cannot send a packet, we'll entirely side-step JAX and immediately return the do-nothing action
      // TODO: If we ever have more than one non-packet action, we should still call into JAX to let the model decide which to do.
      actionIndex = 0;
      VLOG(1) << "Cannot send packet, doing action " << actionIndex;
    }
  }

  // Add the current observation and action to the stack.
  if (pastObservationsAndActions_.size() >= trainingManager_.getPastObservationStackSize()) {
    // We are at capacity, remove the oldest one.
    pastObservationsAndActions_.pop_front();
  }
  pastObservationsAndActions_.emplace_back(observation, actionIndex);

  return actionIndex;
}

sro::scalar_types::ReferenceObjectId DeepLearningIntelligence::avatarHatRefId() const {
  return 23958; // Wizard's Hat (M)
}

} // namespace rl::ai
