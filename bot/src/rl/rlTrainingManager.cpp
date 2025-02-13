#include "bot.hpp"
#include "characterLoginInfo.hpp"
#include "rl/ai/randomIntelligence.hpp"
#include "rl/rlTrainingManager.hpp"
#include "session.hpp"

#include "packet/building/clientAgentCharacterMoveRequest.hpp"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/position_math.hpp>

namespace rl {

RlTrainingManager::RlTrainingManager(const pk2::GameData &gameData,
                  broker::EventBroker &eventBroker,
                  state::WorldState &worldState,
                  ClientManagerInterface &clientManagerInterface) :
                      gameData_(gameData),
                      eventBroker_(eventBroker),
                      worldState_(worldState),
                      clientManagerInterface_(clientManagerInterface)  {
  //
}

void RlTrainingManager::run() {
  // For now, explicitly have two characters fight against each other.
  Session session1(gameData_, eventBroker_, worldState_, clientManagerInterface_);
  Session session2(gameData_, eventBroker_, worldState_, clientManagerInterface_);

  session1.initialize();
  session2.initialize();

  // Explicitly pick which characters we'll use for each session.
  CharacterLoginInfo cli1{/*username=*/"rl0", /*password=*/"0", /*characterName=*/"RL_0"};
  CharacterLoginInfo cli2{/*username=*/"rl1", /*password=*/"0", /*characterName=*/"RL_1"};
  session1.setCharacter(cli1);
  session2.setCharacter(cli2);

  // Start the sessions.
  session1.runAsync();
  session2.runAsync();

  Bot &bot1 = session1.getBot();
  Bot &bot2 = session2.getBot();

  // Intelligence instances exist separate from characters.
  rl::ai::RandomIntelligence randomIntelligence1;
  rl::ai::RandomIntelligence randomIntelligence2;

  // DW South.
  const sro::Position pvpCenterPosition(/*regionId=*/26009,
                                        /*xOffset=*/1221.000,
                                        /*yOffset=*/-101.945,
                                        /*zOffset=*/1809.000);

  auto character1ClientOpenFuture = session1.asyncOpenClient();
  auto character2ClientOpenFuture = session2.asyncOpenClient();
  LOG(INFO) << "Waiting for clients to open";
  character1ClientOpenFuture.wait();
  character2ClientOpenFuture.wait();
  LOG(INFO) << "Clients are open";

  auto character1LoginFuture = bot1.asyncLogIn();
  auto character2LoginFuture = bot2.asyncLogIn();
  LOG(INFO) << "Waiting for characters to log in";
  character1LoginFuture.wait();
  character2LoginFuture.wait();
  LOG(INFO) << "Characters are logged in";

  while (true) {
    // Get the characters ready to fight.
    prepareCharactersForPvp(bot1, bot2, pvpCenterPosition);

    // // Reset the intelligences to prepare them for a new fight. In the future, when these contain RNNs, this will include resetting the memory cells.
    // randomIntelligence1.reset();
    // randomIntelligence2.reset();

    // // Give each character an intelligence.
    // randomIntelligence1.setBot(bot1);
    // randomIntelligence2.setBot(bot2);

    // // Fight.
    // pvp(bot1, bot2);
  }

  // What information do we need here to train the RL agent?
  //  - S,A,R,S
  //    - S - State/observation
  //    - A - Chosen action
  //    - R - Reward
  //    - S - Next state/observation
  // I could publish the observations and actions as events.
  //  The events will need to have which character they're for/from
  //  The observation will need to be decipherable (to compare to the previous and calculate the reward)
}

void RlTrainingManager::prepareCharactersForPvp(Bot &char1, Bot &char2, const sro::Position pvpPosition) {
  // TODO: If the character is dead, resurrect.
  // TODO: If the character is not in pvp mode, enable pvp mode.

  // Move to position (each at a slight offset from the pvp position)
  LOG(INFO) << "Preparing characters for PVP";
  auto character1MoveFuture = char1.asyncMoveTo(sro::position_math::createNewPositionWith2dOffset(pvpPosition, +10.0, 0.0));
  auto character2MoveFuture = char2.asyncMoveTo(sro::position_math::createNewPositionWith2dOffset(pvpPosition, -10.0, 0.0));
  LOG(INFO) << "Moving to " << pvpPosition.toString();
  character1MoveFuture.wait();
  character2MoveFuture.wait();
  LOG(INFO) << "Characters are at " << pvpPosition.toString();

  // TODO: Make sure we have enough potions

  // TODO: Make sure everything is repaired

  while (1) {}
}

void RlTrainingManager::pvp(Bot &char1, Bot &char2) {
  // Start the fight by sending an event.
  eventBroker_.publishEvent(event::EventCode::kRlStartPvp);

  // Wait here until the fight is over.
}

} // namespace rl