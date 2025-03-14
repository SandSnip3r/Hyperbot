#ifndef RL_AI_RANDOM_INTELLIGENCE_HPP_
#define RL_AI_RANDOM_INTELLIGENCE_HPP_

#include "common/random.hpp"
#include "rl/ai/baseIntelligence.hpp"

namespace rl::ai {

class RandomIntelligence : public BaseIntelligence {
public:
  using BaseIntelligence::BaseIntelligence;
  int selectAction(Bot &bot, const Observation &observation, bool canSendPacket) override;
  std::string_view name() const override { return "Random"; }

private:
  std::mt19937 randomEngine_{common::createRandomEngine()};
};

} // namespace rl::ai

#endif // RL_AI_RANDOM_INTELLIGENCE_HPP_