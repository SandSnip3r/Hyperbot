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

// Only one agent may use an intelligence at a time.
class BaseIntelligence {
public:
  BaseIntelligence(TrainingManager &trainingManager);
  virtual void resetForNewEpisode() {}
  virtual int selectAction(Bot &bot, const Observation &observation, bool canSendPacket) = 0;
  virtual const std::string& name() const = 0;
  TrainingManager& trainingManager() { return trainingManager_; }

protected:
  TrainingManager &trainingManager_;
};

} // namespace ai
} // namespace rl

#endif // RL_AI_BASE_INTELLIGENCE_HPP_