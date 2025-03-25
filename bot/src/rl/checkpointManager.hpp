#ifndef RL_CHECKPOINT_MANAGER_HPP_
#define RL_CHECKPOINT_MANAGER_HPP_

#include "rl/jaxInterface.hpp"

#include <ui_proto/rl_checkpointing.pb.h>

#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace broker {
class EventBroker;
} // namespace broker

namespace ui {
class RlUserInterface;
} // namespace ui

namespace rl {

class CheckpointManager {
public:
  CheckpointManager(broker::EventBroker &eventBroker, ui::RlUserInterface &rlUserInterface);
  ~CheckpointManager();
  void saveCheckpoint(const std::string &checkpointName, rl::JaxInterface &jaxInterface, int stepCount);
  bool checkpointExists(const std::string &checkpointName) const;
  std::vector<std::string> getCheckpointNames() const;
private:
  static constexpr std::string_view kCheckpointRegistryFilename{"checkpoint_registry"};
  broker::EventBroker &eventBroker_;
  ui::RlUserInterface &rlUserInterface_;
  mutable std::mutex registryMutex_;
  proto::rl_checkpointing::CheckpointRegistry checkpointRegistry_;
  std::thread checkpointingThread_;
};

} // namespace rl

#endif // RL_CHECKPOINT_MANAGER_HPP_