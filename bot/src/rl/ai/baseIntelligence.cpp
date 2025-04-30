#include "rl/ai/baseIntelligence.hpp"
#include "rl/trainingManager.hpp"

namespace rl::ai {

BaseIntelligence::BaseIntelligence(TrainingManager &trainingManager) : trainingManager_(trainingManager) {

}

} // namespace rl::ai