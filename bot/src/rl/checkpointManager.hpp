#ifndef RL_CHECKPOINT_MANAGER_HPP_
#define RL_CHECKPOINT_MANAGER_HPP_

#include "rl/jaxInterface.hpp"
#include "rl/observationAndActionStorage.hpp"
#include "rl/replayBuffer.hpp"

#include <ui_proto/rl_checkpointing.pb.h>

#include <absl/container/flat_hash_map.h>

#include <mutex>
#include <set>
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
  using ReplayBufferType = ReplayBuffer<ObservationAndActionStorage::Id>;

  void saveCheckpoint(const std::string &checkpointName, JaxInterface &jaxInterface,
                      int stepCount, const ObservationAndActionStorage &observationStorage,
                      const ReplayBufferType &replayBuffer,
                      const absl::flat_hash_map<ObservationAndActionStorage::Id, ReplayBufferType::TransitionId> &observationIdToTransitionIdMap,
                      const absl::flat_hash_map<ReplayBufferType::TransitionId, ObservationAndActionStorage::Id> &transitionIdToObservationIdMap,
                      const std::set<ReplayBufferType::TransitionId> &deletedTransitionIds,
                      std::mutex &replayBufferMutex,
                      bool overwrite);
  bool checkpointExists(const std::string &checkpointName) const;
  std::vector<std::string> getCheckpointNames() const;
  CheckpointValues loadCheckpoint(const std::string &checkpointName, JaxInterface &jaxInterface,
                                  ReplayBufferType &replayBuffer,
                                  ObservationAndActionStorage &observationStorage,
                                  absl::flat_hash_map<ObservationAndActionStorage::Id, ReplayBufferType::TransitionId> &observationIdToTransitionIdMap,
                                  absl::flat_hash_map<ReplayBufferType::TransitionId, ObservationAndActionStorage::Id> &transitionIdToObservationIdMap,
                                  std::set<ReplayBufferType::TransitionId> &deletedTransitionIds,
                                  std::mutex &replayBufferMutex);
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