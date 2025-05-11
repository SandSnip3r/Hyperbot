#ifndef RL_AI_DEEP_LEARNING_INTELLIGENCE_HPP_
#define RL_AI_DEEP_LEARNING_INTELLIGENCE_HPP_

#include "rl/ai/randomIntelligence.hpp"
#include "rl/jaxInterface.hpp"

#include <deque>

namespace rl::ai {

class DeepLearningIntelligence : public RandomIntelligence {
public:
  using RandomIntelligence::RandomIntelligence;
  void resetForNewEpisode() override;
  int selectAction(Bot &bot, const Observation &observation, bool canSendPacket) override;
  const std::string& name() const override { return name_; }
  int getStepCount() const { return stepCount_; }
  void setStepCount(int stepCount) { stepCount_ = stepCount; }
private:
  static constexpr float kInitialEpsilon = 1.0f;
  static constexpr float kFinalEpsilon = 0.01f;
  static constexpr int kEpsilonDecaySteps = 250'000;
  const std::string name_{"DeepLearning"};
  int stepCount_{0};
  float getEpsilon();

  // Newest are at the back, oldest are at the front.
  std::deque<std::pair<Observation, int>> pastObservationsAndActions_;
};

} // namespace rl::ai

#endif // RL_AI_DEEP_LEARNING_INTELLIGENCE_HPP_