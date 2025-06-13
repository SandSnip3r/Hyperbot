#ifndef RL_RL_TRAINING_MANAGER_HPP_
#define RL_RL_TRAINING_MANAGER_HPP_

#include "bot.hpp"
#include "broker/eventBroker.hpp"
#include "clientManagerInterface.hpp"
#include "common/itemRequirement.hpp"
#include "common/pvpDescriptor.hpp"
#include "common/sessionId.hpp"
#include "rl/checkpointManager.hpp"
#include "rl/jaxInterface.hpp"
#include "rl/observationAndActionStorage.hpp"
#include "rl/replayBuffer.hpp"
#include "rl/observation.hpp"
#include "session.hpp"
#include "state/worldState.hpp"
#include "characterLoginInfo.hpp"

#include <silkroad_lib/pk2/gameData.hpp>
#include <silkroad_lib/position.hpp>

#include <absl/base/thread_annotations.h>
#include <absl/container/flat_hash_map.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <utility>
#include <optional>

namespace ui {
class RlUserInterface;
} // namespace ui

namespace rl {

class TrainingManager {
public:
  TrainingManager(const sro::pk2::GameData &gameData,
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
  constexpr int getPastObservationStackSize() const { return kPastObservationStackSize; }
  float getEpsilon() const;

private:
  // Definition for a character pairing at a specific position
  struct CharacterPairing {
    CharacterLoginInfo character1;
    CharacterLoginInfo character2;
    int positionIndex;

    // Track session IDs when characters are active
    std::optional<SessionId> session1Id;
    std::optional<SessionId> session2Id;
  };

  static constexpr int kPastObservationStackSize{16};
  static constexpr float kPvpStartingCenterOffset{40.0f};
  static constexpr int kBatchSize{128};
  static constexpr int kReplayBufferMinimumBeforeTraining{40'000};
  static constexpr int kReplayBufferCapacity{1'000'000};
  static constexpr int kTargetNetworkUpdateInterval{10'000};
  static constexpr int kTrainStepCheckpointInterval{10'000};
  static constexpr float kTargetNetworkPolyakTau{0.0004f};
  static constexpr int kTargetNetworkPolyakUpdateInterval{16};
  static constexpr bool kTargetNetworkPolyakEnabled{true};
  static constexpr float kGamma{0.997f};
  static constexpr float kLearningRate{3e-5f};
  static constexpr float kDropoutRate{0.05f};
  static constexpr float kPerAlpha{0.5f};
  static constexpr float kPerBetaStart{0.4f};
  static constexpr float kPerBetaEnd{1.0f};
  static constexpr int kPerTrainStepCountAnneal{250'000};
  static constexpr float kInitialEpsilon{1.0f};
  static constexpr float kFinalEpsilon{0.01f};
  static constexpr int kEpsilonDecaySteps{200'000};
  static constexpr int kPvpCount{32};

  std::atomic<bool> runTraining_{true};
  std::mutex runTrainingMutex_;
  std::condition_variable runTrainingCondition_;

  const sro::pk2::GameData &gameData_;
  broker::EventBroker &eventBroker_;
  ui::RlUserInterface &rlUserInterface_;
  state::WorldState &worldState_;
  ClientManagerInterface &clientManagerInterface_;
  std::vector<std::unique_ptr<Session>> sessions_;
  std::vector<SessionId> sessionsReadyForAssignment_;
  common::PvpDescriptor::PvpId nextPvpId_{0};
  JaxInterface jaxInterface_{kPastObservationStackSize, kGamma, kLearningRate};
  CheckpointManager checkpointManager_{rlUserInterface_};
  std::atomic<int> trainStepCount_{0};

  // New variables for character pairings and positions
  std::vector<sro::Position> pvpPositions_;
  std::vector<CharacterPairing> characterPairings_;

  // Sample collection rate tracking
  int sampleCount_{0};
  std::chrono::steady_clock::time_point lastSampleTime_{std::chrono::steady_clock::now()};
  static constexpr std::chrono::milliseconds kSampleRateReportInterval{2000};

  // Replay buffer size tracking
  std::chrono::steady_clock::time_point lastReplayBufferSizeUpdateTime_{std::chrono::steady_clock::now()};
  static constexpr std::chrono::milliseconds kReplayBufferSizeUpdateInterval{5000};

  // Training rate tracking
  int trainingCount_{0};
  std::chrono::steady_clock::time_point lastTrainingTime_{std::chrono::steady_clock::now()};
  static constexpr std::chrono::milliseconds kTrainRateReportInterval{2000};

  void precompileModels();

  void defineCharacterPairingsAndPositions();

  void createSessions();
  void train();
  common::PvpDescriptor buildPvpDescriptor(Session &char1, Session &char2, int positionIndex);
  void checkAndPublishPvpDescriptors();
  Session& getSession(SessionId sessionId);

  void pvp(Bot &char1, Bot &char2);

  void buildItemRequirementList();
  std::vector<common::ItemRequirement> itemRequirements_;

  using ReplayBufferType = ReplayBuffer<ObservationAndActionStorage::Id>;
  mutable TracyLockableN(std::mutex, replayBufferAndStorageMutex_, "ReplayBuffer");
  ObservationAndActionStorage observationAndActionStorage_{kReplayBufferCapacity}                                      ABSL_GUARDED_BY(replayBufferAndStorageMutex_);
  ReplayBufferType replayBuffer_{kReplayBufferCapacity, kPerAlpha, /*epsilon=*/1e-5f}                                  ABSL_GUARDED_BY(replayBufferAndStorageMutex_);
  absl::flat_hash_map<ObservationAndActionStorage::Id, ReplayBufferType::TransitionId> observationIdToTransitionIdMap_ ABSL_GUARDED_BY(replayBufferAndStorageMutex_);
  absl::flat_hash_map<ReplayBufferType::TransitionId, ObservationAndActionStorage::Id> transitionIdToObservationIdMap_ ABSL_GUARDED_BY(replayBufferAndStorageMutex_);
  std::set<ReplayBufferType::TransitionId> deletedTransitionIds_                                                       ABSL_GUARDED_BY(replayBufferAndStorageMutex_);
  bool holdingSample_{false}                                                                                           ABSL_GUARDED_BY(replayBufferAndStorageMutex_);

  float calculateReward(const Observation &lastObservation, const Observation &observation, bool isTerminal) const;
  void saveCheckpoint(const std::string &checkpointName, bool overwrite);

  model_inputs::BatchedTrainingInput buildModelInputsFromReplayBufferSamples(const std::vector<ReplayBufferType::SampleResult> &samples) const;
  model_inputs::ModelInputView buildModelInputUpToObservation(ObservationAndActionStorage::Id currentObservationId, model_inputs::BatchedTrainingInput &batchedTrainingInput) const;
};

} // namespace rl

#endif // RL_RL_TRAINING_MANAGER_HPP_
