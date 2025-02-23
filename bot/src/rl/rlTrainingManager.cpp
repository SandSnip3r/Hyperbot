#include "bot.hpp"
#include "characterLoginInfo.hpp"
#include "common/pvpDescriptor.hpp"
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

void RlTrainingManager::createSessions() {
  // For now, explicitly have two characters fight against each other.
  sessions_.push_back(std::make_unique<Session>(gameData_, eventBroker_, worldState_, clientManagerInterface_));
  Session &session1 = *sessions_.back().get();
  sessions_.push_back(std::make_unique<Session>(gameData_, eventBroker_, worldState_, clientManagerInterface_));
  Session &session2 = *sessions_.back().get();

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

  auto character1ClientOpenFuture = session1.asyncOpenClient();
  auto character2ClientOpenFuture = session2.asyncOpenClient();
  LOG(INFO) << "Waiting for clients to open";
  character1ClientOpenFuture.wait();
  character2ClientOpenFuture.wait();
  LOG(INFO) << "Clients are open. Preparing characters for PVP";

  Bot &bot1 = session1.getBot();
  Bot &bot2 = session2.getBot();

  bot1.asyncStandbyForPvp();
  bot2.asyncStandbyForPvp();
}

void RlTrainingManager::run() {
  auto eventHandleFunction = std::bind(&RlTrainingManager::onUpdate, this, std::placeholders::_1);
  // Subscribe to events.
  eventBroker_.subscribeToEvent(event::EventCode::kPvpManagerReadyForAssignment, eventHandleFunction);

  createSessions();

  // bot1.pushAsyncLogIn();
  // bot2.pushAsyncLogIn();
  // bot1.pushAsyncBecomeVisible();
  // bot2.pushAsyncBecomeVisible();
  // std::future<void> character1ReadyForLoop = bot1.pushAsyncEnablePvp();
  // std::future<void> character2ReadyForLoop = bot2.pushAsyncEnablePvp();

  // character1ReadyForLoop.wait();
  // character2ReadyForLoop.wait();
  // LOG(INFO) << "Characters are logged in, visible, and ready for training loop";

  // // DW South.
  // const sro::Position pvpCenterPosition(/*regionId=*/26009,
  //   /*xOffset=*/1221.000,
  //   /*yOffset=*/-101.945,
  //   /*zOffset=*/1809.000);

  // while (true) {
  //   // Get the characters ready to fight.
  //   prepareCharactersForPvp(bot1, bot2, pvpCenterPosition);

  //   // // Reset the intelligences to prepare them for a new fight. In the future, when these contain RNNs, this will include resetting the memory cells.
  //   // randomIntelligence1.reset();
  //   // randomIntelligence2.reset();

  //   // // Give each character an intelligence.
  //   // randomIntelligence1.setBot(bot1);
  //   // randomIntelligence2.setBot(bot2);

  //   // // Fight.
  //   // pvp(bot1, bot2);
  // }

  // // What information do we need here to train the RL agent?
  // //  - S,A,R,S
  // //    - S - State/observation
  // //    - A - Chosen action
  // //    - R - Reward
  // //    - S - Next state/observation
  // // I could publish the observations and actions as events.
  // //  The events will need to have which character they're for/from
  // //  The observation will need to be decipherable (to compare to the previous and calculate the reward)
  while (1) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void RlTrainingManager::onUpdate(const event::Event *event) {
  std::unique_lock worldStateLock(worldState_.mutex);

  LOG(INFO) << "Received event " << event::toString(event->eventCode);
  if (const auto *readyForAssignmentEvent = dynamic_cast<const event::PvpManagerReadyForAssignment*>(event); readyForAssignmentEvent != nullptr) {
    sessionsReadyForAssignment_.push_back(readyForAssignmentEvent->sessionId);
    LOG(INFO) << "Received PvpManagerReadyForAssignment. Now have " << sessionsReadyForAssignment_.size() << " sessions ready for assignment";
    if (sessionsReadyForAssignment_.size() >= 2) {
      createAndPublishPvpDescriptor();
    }
  }
}

void RlTrainingManager::createAndPublishPvpDescriptor() {
  SessionId char1Id = sessionsReadyForAssignment_.at(0);
  SessionId char2Id = sessionsReadyForAssignment_.at(1);
  Session &char1 = getSession(char1Id);
  Session &char2 = getSession(char2Id);

  // ---------------------Build the Pvp Descriptor---------------------
  common::PvpDescriptor pvpDescriptor;
  pvpDescriptor.player1GlobalId = char1.getBot().selfState()->globalId;
  pvpDescriptor.player2GlobalId = char2.getBot().selfState()->globalId;

  // DW South.
  const sro::Position pvpCenterPosition(/*regionId=*/26009,
    /*xOffset=*/1221.000,
    /*yOffset=*/-101.945,
    /*zOffset=*/1809.000);

  pvpDescriptor.pvpPositionPlayer1 = sro::position_math::createNewPositionWith2dOffset(pvpCenterPosition, +kPvpStartingCenterOffset, 0.0);
  pvpDescriptor.pvpPositionPlayer2 = sro::position_math::createNewPositionWith2dOffset(pvpCenterPosition, -kPvpStartingCenterOffset, 0.0);

  pvpDescriptor.itemRequirements = itemRequirements_;

  // TODO: Get ai pointers from the intelligencePool_ and put two into the pvpDescriptor.

  // ----------------------Send the PvpDescriptor----------------------
  eventBroker_.publishEvent<event::BeginPvp>(pvpDescriptor);

  // These two sessions should now no longer be considered for a new Pvp.
  sessionsReadyForAssignment_.erase(sessionsReadyForAssignment_.begin(), sessionsReadyForAssignment_.begin() + 2);
  LOG(INFO) << "Published BeginPvp event for " << char1Id << " and " << char2Id << ". Now have " << sessionsReadyForAssignment_.size() << " sessions ready for assignment";
}

Session& RlTrainingManager::getSession(SessionId sessionId) {
  for (const std::unique_ptr<Session> &sessionPtr : sessions_) {
    if (sessionPtr->sessionId() == sessionId) {
      return *sessionPtr.get();
    }
  }
  throw std::runtime_error("Session not found");
}

void RlTrainingManager::prepareCharactersForPvp(Bot &char1, Bot &char2, const sro::Position pvpPosition) {
  // // TODO: If the character is dead, resurrect.

  // // Move to position (each at a slight offset from the pvp position)
  // LOG(INFO) << "Preparing characters for PVP";
  // LOG(INFO) << "First, moving to " << pvpPosition.toString();
  // char1.pushAsyncMoveTo(sro::position_math::createNewPositionWith2dOffset(pvpPosition, +25.0, 0.0));
  // char2.pushAsyncMoveTo(sro::position_math::createNewPositionWith2dOffset(pvpPosition, -25.0, 0.0));
  // char1.pushAsyncRepair();
  // char2.pushAsyncRepair();

  // // Make sure we have enough potions & other expendables.
  // auto character1PreparationCompleteFuture = char1.pushAsyncMakeSureWeHaveItems(itemRequirements_);
  // auto character2PreparationCompleteFuture = char2.pushAsyncMakeSureWeHaveItems(itemRequirements_);

  // // We have just pushed a bunch of state machines. In case no events are being published, we should send one just to trigger the state machines' onUpdate.
  // eventBroker_.publishEvent(event::EventCode::kDummy);

  // LOG(INFO) << "Waiting on characters to have the required items";
  // character1PreparationCompleteFuture.wait();
  // LOG(INFO) << "First character is done";
  // character2PreparationCompleteFuture.wait();
  // LOG(INFO) << "Characters are at " << pvpPosition.toString() << " and have the required items";
  // // TODO: We also need to ensure that the characters are not invisible.

  // LOG(INFO) << char1.selfState()->name << " pos " << char1.selfState()->position().toString();
  // LOG(INFO) << char2.selfState()->name << " pos " << char2.selfState()->position().toString();

  // while (1) {}
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