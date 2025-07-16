#ifndef RL_AI_DEEP_LEARNING_INTELLIGENCE_HPP_
#define RL_AI_DEEP_LEARNING_INTELLIGENCE_HPP_

#include "rl/ai/randomIntelligence.hpp"
#include "rl/jaxInterface.hpp"

#include <deque>
#include <optional>
#include <string>

namespace rl::ai {

class DeepLearningIntelligence : public RandomIntelligence {
public:
  using RandomIntelligence::RandomIntelligence;
  int selectAction(Bot &bot, const Observation &observation, bool canSendPacket, std::optional<std::string> metadata) override;
  const std::string& name() const override { return name_; }
private:
  const std::string name_{"DeepLearning"};

  // Newest are at the back, oldest are at the front.
  std::deque<std::pair<Observation, int>> pastObservationsAndActions_;
};

} // namespace rl::ai

#endif // RL_AI_DEEP_LEARNING_INTELLIGENCE_HPP_