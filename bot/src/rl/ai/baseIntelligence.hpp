#ifndef RL_AI_BASE_INTELLIGENCE_HPP_
#define RL_AI_BASE_INTELLIGENCE_HPP_

#include "rl/observation.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <memory>
#include <string>

class Bot;

namespace rl {

class TrainingManager;

namespace ai {

// Note: Only one agent may use an intelligence at a time.
class BaseIntelligence {
public:
  BaseIntelligence(TrainingManager &trainingManager);
  TrainingManager& trainingManager() { return trainingManager_; }

  virtual const std::string& name() const = 0;
  virtual int selectAction(Bot &bot, const Observation &observation, bool canSendPacket, std::optional<std::string> metadata = std::nullopt) = 0;
  virtual sro::scalar_types::ReferenceObjectId avatarHatRefId() const = 0;

protected:
  TrainingManager &trainingManager_;
};

} // namespace ai
} // namespace rl

#endif // RL_AI_BASE_INTELLIGENCE_HPP_