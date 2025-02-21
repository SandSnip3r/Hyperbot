#ifndef RL_RL_TRAINING_MANAGER_HPP_
#define RL_RL_TRAINING_MANAGER_HPP_

#include "bot.hpp"
#include "broker/eventBroker.hpp"
#include "clientManagerInterface.hpp"
#include "common/itemRequirement.hpp"
#include "pk2/gameData.hpp"
#include "rl/intelligencePool.hpp"
#include "session.hpp"
#include "common/sessionId.hpp"
#include "state/worldState.hpp"

#include <silkroad_lib/position.hpp>

#include <memory>
#include <vector>

namespace rl {

class RlTrainingManager {
public:
  RlTrainingManager(const pk2::GameData &gameData,
                    broker::EventBroker &eventBroker,
                    state::WorldState &worldState,
                    ClientManagerInterface &clientManagerInterface);

  // Blocks.
  void run();

  void onUpdate(const event::Event *event);
private:
  static constexpr float kPvpStartingCenterOffset{30.0f};
  const pk2::GameData &gameData_;
  broker::EventBroker &eventBroker_;
  state::WorldState &worldState_;
  ClientManagerInterface &clientManagerInterface_;
  std::vector<std::unique_ptr<Session>> sessions_;
  std::vector<SessionId> sessionsReadyForAssignment_;
  IntelligencePool intelligencePool_;

  void createSessions();
  void createAndPublishPvpDescriptor();
  Session& getSession(SessionId sessionId);

  void prepareCharactersForPvp(Bot &char1, Bot &char2, const sro::Position pvpPosition);
  void pvp(Bot &char1, Bot &char2);

  void buildItemRequirementList();
  std::vector<common::ItemRequirement> itemRequirements_;
};

} // namespace rl

#endif // RL_RL_TRAINING_MANAGER_HPP_