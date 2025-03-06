#ifndef RL_AI_RANDOM_INTELLIGENCE_HPP_
#define RL_AI_RANDOM_INTELLIGENCE_HPP_

#include "common/random.hpp"
#include "rl/ai/baseIntelligence.hpp"

namespace rl::ai {

class RandomIntelligence : public BaseIntelligence {
public:
  using BaseIntelligence::BaseIntelligence;
  std::unique_ptr<Action> selectAction(Bot &bot, const event::Event *event, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId) override;

private:
  std::mt19937 randomEngine_{common::createRandomEngine()};
};

} // namespace rl::ai

#endif // RL_AI_RANDOM_INTELLIGENCE_HPP_