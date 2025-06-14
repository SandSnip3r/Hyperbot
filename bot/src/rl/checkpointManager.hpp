#ifndef RL_CHECKPOINT_MANAGER_HPP_
#define RL_CHECKPOINT_MANAGER_HPP_

#include "rl/jaxInterface.hpp"

#include <ui_proto/rl_checkpointing.pb.h>

#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ui {
class RlUserInterface;
} // namespace ui

namespace rl {

struct CheckpointValues {
  int stepCount;
};

class CheckpointManager {
public:
  CheckpointManager(ui::RlUserInterface &rlUserInterface);
  ~CheckpointManager();
  void saveCheckpoint(const std::string &checkpointName, rl::JaxInterface &jaxInterface, int stepCount, bool overwrite);
  bool checkpointExists(const std::string &checkpointName) const;
  std::vector<std::string> getCheckpointNames() const;
  CheckpointValues loadCheckpoint(const std::string &checkpointName, rl::JaxInterface &jaxInterface);
  void deleteCheckpoints(const std::vector<std::string> &checkpointNames);
private:
  static constexpr std::string_view kCheckpointRegistryFilename{"checkpoint_registry"};
  static constexpr std::string_view kCheckpointDirectoryName{"checkpoints"};
  ui::RlUserInterface &rlUserInterface_;
  mutable std::mutex registryMutex_;
  proto::rl_checkpointing::CheckpointRegistry checkpointRegistry_;
  std::thread checkpointingThread_;

  // Note: Expects lock to be held.
  void saveCurrentRegistryNoLock();
};

} // namespace rl

#endif // RL_CHECKPOINT_MANAGER_HPP_