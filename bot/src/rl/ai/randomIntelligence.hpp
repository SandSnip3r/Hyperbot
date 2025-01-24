#ifndef RL_AI_RANDOM_INTELLIGENCE_HPP_
#define RL_AI_RANDOM_INTELLIGENCE_HPP_

#include "rl/ai/baseIntelligence.hpp"

namespace rl::ai {

class RandomIntelligence : public BaseIntelligence {
public:
  void onUpdate(event::Event *event) override;
  void reset() override;
};

} // namespace rl::ai

#endif // RL_AI_RANDOM_INTELLIGENCE_HPP_