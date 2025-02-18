#include "bot.hpp"
#include "characterLoginInfo.hpp"
#include "rl/ai/randomIntelligence.hpp"
#include "rl/rlTrainingManager.hpp"
#include "session.hpp"
#include "type_id/categories.hpp"

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
  buildItemRequirementList();
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

  LOG(INFO) << "Logging in characters";
  bot1.pushAsyncLogIn();
  bot2.pushAsyncLogIn();

  bot1.pushAsyncBecomeVisible();
  bot2.pushAsyncBecomeVisible();

  std::future<void> character1ReadyForLoop = bot1.pushAsyncEnablePvp();
  std::future<void> character2ReadyForLoop = bot2.pushAsyncEnablePvp();
  character1ReadyForLoop.wait();
  character2ReadyForLoop.wait();
  LOG(INFO) << "Characters logged in, visible, and ready for training loop";

  // auto character1VisibleFuture = bot1.asyncBecomeVisible();
  // auto character2VisibleFuture = bot2.asyncBecomeVisible();
  // LOG(INFO) << "Waiting for characters to be visible";
  // character1VisibleFuture.wait();
  // character2VisibleFuture.wait();
  // LOG(INFO) << "Characters are visible";

  // TODO: Ensure PVP mode is enabled.

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

  // Move to position (each at a slight offset from the pvp position)
  LOG(INFO) << "Preparing characters for PVP";
  LOG(INFO) << "First, moving to " << pvpPosition.toString();
  char1.pushAsyncMoveTo(sro::position_math::createNewPositionWith2dOffset(pvpPosition, +25.0, 0.0));
  char2.pushAsyncMoveTo(sro::position_math::createNewPositionWith2dOffset(pvpPosition, -25.0, 0.0));

  // Make sure we have enough potions & other expendables.
  auto character1PreparationCompleteFuture = char1.pushAsyncMakeSureWeHaveItems(itemRequirements_);
  auto character2PreparationCompleteFuture = char2.pushAsyncMakeSureWeHaveItems(itemRequirements_);
  LOG(INFO) << "Waiting on characters to have the required items";
  character1PreparationCompleteFuture.wait();
  LOG(INFO) << "First character is done";
  character2PreparationCompleteFuture.wait();
  LOG(INFO) << "Characters are at " << pvpPosition.toString() << " and have the required items";
  // TODO: We also need to ensure that the characters are not invisible.

  LOG(INFO) << char1.selfState()->name << " pos " << char1.selfState()->position().toString();
  LOG(INFO) << char2.selfState()->name << " pos " << char2.selfState()->position().toString();

  // TODO: Make sure everything is repaired

  while (1) {}
}

void RlTrainingManager::pvp(Bot &char1, Bot &char2) {
  // Start the fight by sending an event.
  eventBroker_.publishEvent(event::EventCode::kRlStartPvp);

  // Wait here until the fight is over.
}

void RlTrainingManager::buildItemRequirementList() {
  const sro::pk2::ref::ItemId smallHpPotionRefId = gameData_.itemData().getItemId([](const sro::pk2::ref::Item &item) {
    return type_id::categories::kHpPotion.contains(type_id::getTypeId(item)) && item.itemClass == 2;
  });
  const sro::pk2::ref::ItemId smallMpPotionRefId = gameData_.itemData().getItemId([](const sro::pk2::ref::Item &item) {
    return type_id::categories::kMpPotion.contains(type_id::getTypeId(item)) && item.itemClass == 2;
  });
  const sro::pk2::ref::ItemId mediumUniversalPillRefId = gameData_.itemData().getItemId([](const sro::pk2::ref::Item &item) {
    return type_id::categories::kUniversalPill.contains(type_id::getTypeId(item)) && item.itemClass == 2;
  });

  constexpr int kSmallHpPotionRequiredCount = 200;
  constexpr int kSmallMpPotionRequiredCount = 200;
  constexpr int kMediumUniversalPillRequiredCount = 100;
  itemRequirements_.push_back({smallHpPotionRefId, kSmallHpPotionRequiredCount});
  itemRequirements_.push_back({smallMpPotionRefId, kSmallMpPotionRequiredCount});
  itemRequirements_.push_back({mediumUniversalPillRefId, kMediumUniversalPillRequiredCount});
}

} // namespace rl