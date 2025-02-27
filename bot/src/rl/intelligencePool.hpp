#ifndef RL_INTELLIGENCE_POOL_HPP_
#define RL_INTELLIGENCE_POOL_HPP_

#include "rl/ai/baseIntelligence.hpp"
#include "rl/ai/randomIntelligence.hpp"

namespace rl {

class IntelligencePool {
public:
  ai::RandomIntelligence* getRandomIntelligence() {
    return &randomIntelligence_;
  }
private:
  ai::RandomIntelligence randomIntelligence_;
};

} // namespace rl

#endif // RL_INTELLIGENCE_POOL_HPP_