#ifndef RL_AI_NO_OP_INTELLIGENCE_HPP_
#define RL_AI_NO_OP_INTELLIGENCE_HPP_

#include "common/random.hpp"
#include "rl/ai/baseIntelligence.hpp"

#include <optional>

namespace rl::ai {

class NoOpIntelligence : public BaseIntelligence {
public:
  using BaseIntelligence::BaseIntelligence;
  const std::string& name() const override { return name_; }
  int selectAction(Bot &bot, const Observation &observation, bool canSendPacket, std::optional<std::string> metadata = std::nullopt) override { return 0; }
  sro::scalar_types::ReferenceObjectId avatarHatRefId() const override;

protected:
  const std::string name_{"No-Op"};
  std::mt19937 randomEngine_{common::createRandomEngine()};
};

} // namespace rl::ai

#endif // RL_AI_NO_OP_INTELLIGENCE_HPP_