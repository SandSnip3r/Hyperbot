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
#include "rl/jaxInterface.hpp"
#include "rl/observationAndActionStorage.hpp"
#include "rl/replayBuffer.hpp"
#include "rl/observation.hpp"
#include "session.hpp"
#include "state/worldState.hpp"
#include "characterLoginInfo.hpp"

#include <silkroad_lib/position.hpp>

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

  static constexpr int kPastObservationStackSize{64};
  static constexpr float kPvpStartingCenterOffset{40.0f};
  static constexpr int kBatchSize{128};
  static constexpr int kReplayBufferMinimumBeforeTraining{10'000};
  static constexpr int kReplayBufferCapacity{1'000'000};
  static constexpr int kTargetNetworkUpdateInterval{10'000};
  static constexpr int kTrainStepCheckpointInterval{10'000};
  static constexpr float kTargetNetworkPolyakTau{0.0005f};
  static constexpr int kTargetNetworkPolyakUpdateInterval{16};
  static constexpr bool kTargetNetworkPolyakEnabled{true};
  static constexpr float kGamma{0.99f};
  static constexpr float kLearningRate{1e-6f};
  static constexpr float kDropoutRate{0.1f};
  static constexpr float kPerAlpha{0.5f};
  static constexpr float kPerBetaStart{0.4f};
  static constexpr float kPerBetaEnd{1.0f};
  static constexpr int kPerTrainStepCountAnneal{250'000};
  static constexpr float kInitialEpsilon{1.0f};
  static constexpr float kFinalEpsilon{0.01f};
  static constexpr int kEpsilonDecaySteps{250'000};
  static constexpr int kPvpCount{4};

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
  JaxInterface jaxInterface_{kPastObservationStackSize, kGamma, kLearningRate};
  CheckpointManager checkpointManager_{rlUserInterface_};
  std::atomic<int> trainStepCount_{0};
  std::atomic<int> actionStepCount_{0};

  // New variables for character pairings and positions
  std::vector<sro::Position> pvpPositions_;
  std::vector<CharacterPairing> characterPairings_;

  // Sample collection rate tracking
  int sampleCount_{0};
  std::chrono::high_resolution_clock::time_point lastSampleTime_{std::chrono::high_resolution_clock::now()};
  static constexpr std::chrono::milliseconds kSampleRateReportInterval{2000};

  // Replay buffer size tracking
  std::chrono::high_resolution_clock::time_point lastReplayBufferSizeUpdateTime_{std::chrono::high_resolution_clock::now()};
  static constexpr std::chrono::milliseconds kReplayBufferSizeUpdateInterval{5000};

  // Training rate tracking
  int trainingCount_{0};
  std::chrono::high_resolution_clock::time_point lastTrainingTime_{std::chrono::high_resolution_clock::now()};
  static constexpr std::chrono::milliseconds kTrainRateReportInterval{2000};

  void setUpIntelligencePool();
  void defineCharacterPairingsAndPositions();

  void createSessions();
  void train();
  common::PvpDescriptor buildPvpDescriptor(Session &char1, Session &char2, int positionIndex);
  void checkAndPublishPvpDescriptors();
  Session& getSession(SessionId sessionId);

  void pvp(Bot &char1, Bot &char2);

  void buildItemRequirementList();
  std::vector<common::ItemRequirement> itemRequirements_;

  ObservationAndActionStorage observationAndActionStorage_{kReplayBufferCapacity};
  using ReplayBufferType = ReplayBuffer<ObservationAndActionStorage::Id>;
  ReplayBufferType replayBuffer_{kReplayBufferCapacity, kPerAlpha, /*epsilon=*/1e-5f};
  absl::flat_hash_map<ObservationAndActionStorage::Id, ReplayBufferType::TransitionId> observationIdToTransitionIdMap_;
  absl::flat_hash_map<ReplayBufferType::TransitionId, ObservationAndActionStorage::Id> transitionIdToObservationIdMap_;
  mutable std::mutex observationTransitionIdMapMutex_;
  float calculateReward(const Observation &lastObservation, const Observation &observation, bool isTerminal) const;
  void saveCheckpoint(const std::string &checkpointName, bool overwrite);

  struct ModelInputs {
    std::vector<ModelInput> oldModelInputs;
    std::vector<int> actionsTaken;
    std::vector<bool> isTerminals;
    std::vector<float> rewards;
    std::vector<ModelInput> newModelInputs;
    std::vector<float> importanceSamplingWeights;
  };

  ModelInputs buildModelInputsFromReplayBufferSamples(const std::vector<ReplayBufferType::SampleResult> &samples) const;
  ModelInput buildModelInputUpToObservation(ObservationAndActionStorage::Id currentObservationId) const;
};

} // namespace rl

#endif // RL_RL_TRAINING_MANAGER_HPP_