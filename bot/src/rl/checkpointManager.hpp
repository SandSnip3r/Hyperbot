#ifndef RL_CHECKPOINT_MANAGER_HPP_
#define RL_CHECKPOINT_MANAGER_HPP_

#include <string>
#include <vector>

namespace rl {

class CheckpointManager {
public:
  void saveCheckpoint(const std::string &checkpointName);
  bool checkpointExists(const std::string &checkpointName) const;
  std::vector<std::string> getCheckpointNames() const;
private:
};

} // namespace rl

#endif // RL_CHECKPOINT_MANAGER_HPP_