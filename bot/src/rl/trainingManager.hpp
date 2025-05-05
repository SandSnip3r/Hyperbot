#ifndef RL_RL_TRAINING_MANAGER_HPP_
#define RL_RL_TRAINING_MANAGER_HPP_

#include "bot.hpp"
#include "broker/eventBroker.hpp"
#include "clientManagerInterface.hpp"
#include "common/itemRequirement.hpp"
#include "common/pvpDescriptor.hpp"
#include "common/sessionId.hpp"
#include "pk2/gameData.hpp"
#include "rl/checkpointManager.hpp"
#include "rl/intelligencePool.hpp"
#include "rl/jaxInterface.hpp"
#include "rl/observationAndActionStorage.hpp"
#include "rl/replayBuffer.hpp"
#include "rl/observation.hpp"
#include "session.hpp"
#include "state/worldState.hpp"

#include <silkroad_lib/position.hpp>

#include <absl/container/flat_hash_map.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>

namespace ui {
class RlUserInterface;
} // namespace ui

namespace rl {

class TrainingManager {
public:
  TrainingManager(const pk2::GameData &gameData,
                    broker::EventBroker &eventBroker,
                    ui::RlUserInterface &rlUserInterface,
                    state::WorldState &worldState,
                    ClientManagerInterface &clientManagerInterface);

  // Blocks.
  void run();

  void onUpdate(const event::Event *event);

  // If the actionIndex is not set, the observation was terminal.
  void reportObservationAndAction(common::PvpDescriptor::PvpId pvpId, const std::string &intelligenceName, const Observation &observation, std::optional<int> actionIndex);

  JaxInterface& getJaxInterface() { return jaxInterface_; }
  int getTrainStepCount() const { return trainStepCount_.load(); }
  constexpr int getObservationStackSize() const { return kObservationStackSize; }

private:
  static constexpr int kObservationStackSize = 64;
  static constexpr float kPvpStartingCenterOffset{40.0f};
  static constexpr int kBatchSize{128};
  static constexpr int kReplayBufferCapacity{1'000'000};
  static constexpr int kTargetNetworkUpdateInterval{10'000};
  static constexpr int kTrainStepCheckpointInterval{10'000};
  static constexpr float kTargetNetworkPolyakTau{0.0005f};
  static constexpr bool kUsePolyakTargetNetworkUpdate{false};
  static constexpr float kGamma{0.99f};
  static constexpr float kLearningRate{3e-6f};

  std::atomic<bool> runTraining_{true};
  std::mutex runTrainingMutex_;
  std::condition_variable runTrainingCondition_;

  const pk2::GameData &gameData_;
  broker::EventBroker &eventBroker_;
  ui::RlUserInterface &rlUserInterface_;
  state::WorldState &worldState_;
  ClientManagerInterface &clientManagerInterface_;
  std::vector<std::unique_ptr<Session>> sessions_;
  std::vector<SessionId> sessionsReadyForAssignment_;
  common::PvpDescriptor::PvpId nextPvpId_{0};
  IntelligencePool intelligencePool_{*this};
  JaxInterface jaxInterface_{kGamma, kLearningRate};
  CheckpointManager checkpointManager_{rlUserInterface_};
  std::atomic<int> trainStepCount_{0};

  // Sample collection rate tracking
  int sampleCount_{0};
  std::chrono::high_resolution_clock::time_point lastSampleTime_{std::chrono::high_resolution_clock::now()};

  // Training rate tracking
  int trainingCount_{0};
  std::chrono::high_resolution_clock::time_point lastTrainingTime_{std::chrono::high_resolution_clock::now()};

  void setUpIntelligencePool();

  void createSessions();
  void train();
  common::PvpDescriptor buildPvpDescriptor(Session &char1, Session &char2);
  void createAndPublishPvpDescriptor();
  Session& getSession(SessionId sessionId);

  void pvp(Bot &char1, Bot &char2);

  void buildItemRequirementList();
  std::vector<common::ItemRequirement> itemRequirements_;

  ObservationAndActionStorage observationAndActionStorage_{kReplayBufferCapacity};
  using ReplayBufferType = ReplayBuffer<ObservationAndActionStorage::Id>;
  ReplayBufferType replayBuffer_{kReplayBufferCapacity, /*alpha=*/0.6f, /*beta=*/0.8f, /*epsilon=*/1e-5f};
  absl::flat_hash_map<ObservationAndActionStorage::Id, ReplayBufferType::TransitionId> observationIdToTransitionIdMap_;
  absl::flat_hash_map<ReplayBufferType::TransitionId, ObservationAndActionStorage::Id> transitionIdToObservationIdMap_;
  mutable std::mutex observationTransitionIdMapMutex_;
  float calculateReward(const Observation &lastObservation, const Observation &observation, bool isTerminal) const;
  void saveCheckpoint(const std::string &checkpointName, bool overwrite);

  struct ModelInputs {
    std::vector<std::vector<Observation>> olderObservationStacks;
    std::vector<int> actionIndexs;
    std::vector<bool> isTerminals;
    std::vector<float> rewards;
    std::vector<std::vector<Observation>> newerObservationStacks;
    std::vector<float> weights;
  };

  ModelInputs buildModelInputsFromReplayBufferSamples(const std::vector<ReplayBufferType::SampleResult> &samples) const;
};

} // namespace rl

#endif // RL_RL_TRAINING_MANAGER_HPP_