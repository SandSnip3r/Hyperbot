#include "bot.hpp"
#include "characterLoginInfo.hpp"
#include "rl/ai/randomIntelligence.hpp"
#include "rl/rlTrainingManager.hpp"
#include "session.hpp"

#include "packet/building/clientAgentCharacterMoveRequest.hpp"

#include <silkroad_lib/position.hpp>

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
  // Session session2(gameData_, eventBroker_, worldState_, clientManagerInterface_);

  session1.initialize();
  // session2.initialize();

  // Explicitly pick which characters we'll use for each session.
  CharacterLoginInfo cli1{/*username=*/"rl0", /*password=*/"0", /*characterName=*/"RL_0"};
  CharacterLoginInfo cli2{/*username=*/"rl1", /*password=*/"0", /*characterName=*/"RL_1"};
  session1.setCharacter(cli1);
  // session2.setCharacter(cli2);

  // Start the sessions.
  session1.runAsync();
  // session2.runAsync();

  Bot &bot1 = session1.getBot();
  // Bot &bot2 = session2.getBot();

  // Intelligence instances exist separate from characters.
  rl::ai::RandomIntelligence randomIntelligence1;
  rl::ai::RandomIntelligence randomIntelligence2;

  // DW South.
  const sro::Position pvpCenterPosition(/*regionId=*/26009,
                                        /*xOffset=*/1221.000,
                                        /*yOffset=*/-101.945,
                                        /*zOffset=*/1809.000);

  auto character1Future = bot1.getFutureForClientOpening();
  LOG(INFO) << "Waiting for client to open";
  character1Future.wait();
  LOG(INFO) << "Client is open";

  while (true) {
    // // Get the characters ready to fight.
    // prepareCharactersForPvp(bot1, bot2, pvpCenterPosition);

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
  std::future<void> character1Future;
  std::future<void> character2Future;

  // If the client is not open, wait for it.
  //  TODO: Also open it.


  // If the character is not logged in, login.
  if (!char1.loggedIn()) {
    LOG(INFO) << "character 1 is not logged in. logging in";
    character1Future = char1.logIn();
  }
  if (character1Future.valid()) {
    LOG(INFO) << "Waiting for character 1 to log in";
    character1Future.wait();
    LOG(INFO) << "Character 1 is logged in";
  }
  if (!char2.loggedIn()) {
    LOG(INFO) << "character 2 is not logged in. logging in";
    character2Future = char2.logIn();
  }
  if (character2Future.valid()) {
    LOG(INFO) << "Waiting for character 2 to log in";
    character2Future.wait();
    LOG(INFO) << "Character 2 is logged in";
  }
  LOG(INFO) << "Both characters are logged in";

  // TODO: If the character is dead, resurrect.
  // TODO: If the character is not in pvp mode, enable pvp mode.

  // Make sure we have enough potions
  // Make sure everything is repaired
  // Move to position (each at a slight offset from the pvp position)
  LOG(INFO) << "Preparing characters for PVP";
  LOG(INFO) << "Moving to " << pvpPosition.toString();
  PacketContainer movePacket = packet::building::ClientAgentCharacterMoveRequest::moveToPosition(pvpPosition);
  char1.packetBroker().injectPacket(movePacket, PacketContainer::Direction::kClientToServer);
  char2.packetBroker().injectPacket(movePacket, PacketContainer::Direction::kClientToServer);
  while (1) {}
}

void RlTrainingManager::pvp(Bot &char1, Bot &char2) {
  // Start the fight by sending an event.
  eventBroker_.publishEvent(event::EventCode::kRlStartPvp);

  // Wait here until the fight is over.
}

} // namespace rl