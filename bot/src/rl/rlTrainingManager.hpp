#ifndef RL_RL_TRAINING_MANAGER_HPP_
#define RL_RL_TRAINING_MANAGER_HPP_

#include "bot.hpp"
#include "broker/eventBroker.hpp"
#include "clientManagerInterface.hpp"
#include "pk2/gameData.hpp"
#include "state/worldState.hpp"

#include <silkroad_lib/position.hpp>

namespace rl {

class RlTrainingManager {
public:
  RlTrainingManager(const pk2::GameData &gameData,
                    broker::EventBroker &eventBroker,
                    state::WorldState &worldState,
                    ClientManagerInterface &clientManagerInterface);

  // Blocks.
  void run();
private:
  const pk2::GameData &gameData_;
  broker::EventBroker &eventBroker_;
  state::WorldState &worldState_;
  ClientManagerInterface &clientManagerInterface_;
  void prepareCharactersForPvp(Bot &char1, Bot &char2, const sro::Position pvpPosition);
  void pvp(Bot &char1, Bot &char2);

  void buildItemRequirementList();
  std::vector<Bot::ItemRequirement> itemRequirements_;
};

} // namespace rl

#endif // RL_RL_TRAINING_MANAGER_HPP_