#ifndef RL_AI_BASE_INTELLIGENCE_HPP_
#define RL_AI_BASE_INTELLIGENCE_HPP_

#include "common/pvpDescriptor.hpp"
#include "rl/action.hpp"
#include "rl/observation.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <memory>
#include <string>

namespace event {
struct Event;
} // namespace event

class Bot;

namespace rl {

class TrainingManager;

namespace ai {

class BaseIntelligence {
public:
  BaseIntelligence(TrainingManager &trainingManager) : trainingManager_(trainingManager) {}
  virtual int selectAction(Bot &bot, const Observation &observation, bool canSendPacket) = 0;
  virtual const std::string& name() const = 0;
  TrainingManager& trainingManager() { return trainingManager_; }

protected:
  TrainingManager &trainingManager_;
};

} // namespace ai
} // namespace rl

#endif // RL_AI_BASE_INTELLIGENCE_HPP_