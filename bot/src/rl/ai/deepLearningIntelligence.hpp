#ifndef RL_AI_DEEP_LEARNING_INTELLIGENCE_HPP_
#define RL_AI_DEEP_LEARNING_INTELLIGENCE_HPP_

#include "rl/ai/baseIntelligence.hpp"
#include "rl/jaxInterface.hpp"

namespace rl::ai {

class DeepLearningIntelligence : public BaseIntelligence {
public:
  using BaseIntelligence::BaseIntelligence;
  std::unique_ptr<Action> selectAction(Bot &bot, const event::Event *event, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId) override;
  std::string_view name() const override { return "DeepLearning"; }
private:
};

} // namespace rl::ai

#endif // RL_AI_DEEP_LEARNING_INTELLIGENCE_HPP_