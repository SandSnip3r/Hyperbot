#ifndef RL_RL_TRAINING_MANAGER_HPP_
#define RL_RL_TRAINING_MANAGER_HPP_

#include "bot.hpp"
#include "broker/eventBroker.hpp"
#include "clientManagerInterface.hpp"
#include "common/itemRequirement.hpp"
#include "common/pvpDescriptor.hpp"
#include "common/sessionId.hpp"
#include "pk2/gameData.hpp"
#include "rl/intelligencePool.hpp"
#include "rl/jaxInterface.hpp"
#include "rl/observation.hpp"
#include "session.hpp"
#include "state/worldState.hpp"

#include <silkroad_lib/position.hpp>

#include <memory>
#include <vector>

namespace rl {

class TrainingManager {
public:
  TrainingManager(const pk2::GameData &gameData,
                    broker::EventBroker &eventBroker,
                    state::WorldState &worldState,
                    ClientManagerInterface &clientManagerInterface);

  // Blocks.
  void run();

  void onUpdate(const event::Event *event);

  // If the actionIndex is not set, the observation was terminal.
  void reportEventObservationAndAction(common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId observerGlobalId, const event::Event *event, const Observation &observation, std::optional<int> actionIndex);

  JaxInterface& getJaxInterface() { return jaxInterface_; }

private:
  static constexpr bool kTrain{true};
  static constexpr float kPvpStartingCenterOffset{40.0f};
  std::atomic<bool> training_{true};
  const pk2::GameData &gameData_;
  broker::EventBroker &eventBroker_;
  state::WorldState &worldState_;
  ClientManagerInterface &clientManagerInterface_;
  std::vector<std::unique_ptr<Session>> sessions_;
  std::vector<SessionId> sessionsReadyForAssignment_;
  common::PvpDescriptor::PvpId nextPvpId_{0};
  IntelligencePool intelligencePool_{*this};
  JaxInterface jaxInterface_;
  int trainStepCount_{0};
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

  struct ReplayBufferEntry {
    Observation observation;
    int actionIndex;
    double reward;
    Observation nextObservation;
  };

  struct LastObservationAndAction {
    Observation observation;
    int actionIndex;
  };

  std::vector<ReplayBufferEntry> replayBuffer_;
  std::map<sro::scalar_types::EntityGlobalId, LastObservationAndAction> lastObservationMap_;
  std::map<common::PvpDescriptor::PvpId, std::map<sro::scalar_types::EntityGlobalId, std::vector<std::tuple<event::EventCode, Observation, std::optional<int>>>>> newReplayBuffer_;
  std::mutex replayBufferMutex_;
  double calculateReward(const Observation &lastObservation, const Observation &observation) const;
};

} // namespace rl

#endif // RL_RL_TRAINING_MANAGER_HPP_