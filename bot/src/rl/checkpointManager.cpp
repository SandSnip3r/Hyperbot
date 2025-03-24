#include "checkpointManager.hpp"

namespace rl {

void CheckpointManager::saveCheckpoint(const std::string &checkpointName) {

}

bool CheckpointManager::checkpointExists(const std::string &checkpointName) const {
  return true;
}

std::vector<std::string> CheckpointManager::getCheckpointNames() const {
  return {"test123"};
}

} // namespace rl