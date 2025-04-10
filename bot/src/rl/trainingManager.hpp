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
#include "rl/replayBuffer.hpp"
#include "rl/observation.hpp"
#include "session.hpp"
#include "state/worldState.hpp"

#include <silkroad_lib/position.hpp>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

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
  void reportObservationAndAction(common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId observerGlobalId, const Observation &observation, std::optional<int> actionIndex);

  JaxInterface& getJaxInterface() { return jaxInterface_; }
  int getTrainStepCount() const { return trainStepCount_.load(); }

private:
  static constexpr bool kTrain{true};
  static constexpr float kPvpStartingCenterOffset{40.0f};

  std::atomic<bool> runTraining_{false};
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
  JaxInterface jaxInterface_;
  CheckpointManager checkpointManager_{rlUserInterface_};
  std::atomic<int> trainStepCount_{0};
  static constexpr int kTargetNetworkUpdateInterval{10000};

  void setUpIntelligencePool();

  void createSessions();
  void train();
  common::PvpDescriptor buildPvpDescriptor(Session &char1, Session &char2);
  void createAndPublishPvpDescriptor();
  Session& getSession(SessionId sessionId);

  void pvp(Bot &char1, Bot &char2);

  void buildItemRequirementList();
  std::vector<common::ItemRequirement> itemRequirements_;

  ReplayBuffer replayBuffer_{/*capacity=*/10'000'000, /*samplingBatchSize=*/1,
                             /*alpha=*/0.6f, /*beta=*/0.4f, /*epsilon=*/1e-5f};
  float calculateReward(const Observation &lastObservation, const Observation &observation) const;
  void saveCheckpoint(const std::string &checkpointName);
};

} // namespace rl

#endif // RL_RL_TRAINING_MANAGER_HPP_