#ifndef RL_INTELLIGENCE_POOL_HPP_
#define RL_INTELLIGENCE_POOL_HPP_

#include "rl/ai/baseIntelligence.hpp"
#include "rl/ai/randomIntelligence.hpp"

namespace rl {

class TrainingManager;

class IntelligencePool {
public:
  IntelligencePool(TrainingManager &trainingManager) : trainingManager_(trainingManager) {}
  ai::RandomIntelligence* getRandomIntelligence() {
    return &randomIntelligence_;
  }
private:
  TrainingManager &trainingManager_;
  ai::RandomIntelligence randomIntelligence_{trainingManager_};
};

} // namespace rl

#endif // RL_INTELLIGENCE_POOL_HPP_