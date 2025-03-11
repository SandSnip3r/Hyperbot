#include "bot.hpp"
#include "characterLoginInfo.hpp"
#include "common/pvpDescriptor.hpp"
#include "rl/ai/randomIntelligence.hpp"
#include "rl/trainingManager.hpp"
#include "session.hpp"
#include "type_id/categories.hpp"

#include "packet/building/clientAgentCharacterMoveRequest.hpp"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/position_math.hpp>

namespace rl {

TrainingManager::TrainingManager(const pk2::GameData &gameData,
                  broker::EventBroker &eventBroker,
                  state::WorldState &worldState,
                  ClientManagerInterface &clientManagerInterface) :
                      gameData_(gameData),
                      eventBroker_(eventBroker),
                      worldState_(worldState),
                      clientManagerInterface_(clientManagerInterface)  {
  buildItemRequirementList();
}

void TrainingManager::run() {
  setUpIntelligencePool();

  auto eventHandleFunction = std::bind(&TrainingManager::onUpdate, this, std::placeholders::_1);
  // Subscribe to events.
  eventBroker_.subscribeToEvent(event::EventCode::kPvpManagerReadyForAssignment, eventHandleFunction);

  jaxInterface_.initialize();

  createSessions();

  train();
}

void TrainingManager::train() {
  // Wait until we have enough samples to start training.
  // while (1) {
  //   std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // }
  // We now have enough samples to start training.
  // Train at full speed in a tight loop.
  while (1) {
    // Get a S,A,R,S' tuple from the replay buffer.
    if (!newReplayBuffer_.empty()) {
      auto eng = common::createRandomEngine();
      std::uniform_int_distribution<int> dist(0, newReplayBuffer_.size()-1);
      auto it = newReplayBuffer_.begin();
      std::advance(it, dist(eng));
      const common::PvpDescriptor::PvpId pvpId = it->first;
      const auto &pvpMap = it->second;
      if (!pvpMap.empty()) {
        std::uniform_int_distribution<int> dist2(0, pvpMap.size()-1);
        auto it2 = pvpMap.begin();
        std::advance(it2, dist2(eng));
        const sro::scalar_types::EntityGlobalId observerGlobalId = it2->first;
        const auto &eventObservationActionList = it2->second;
        if (eventObservationActionList.size() >= 2) {
          std::uniform_int_distribution<int> dist3(1, eventObservationActionList.size()-1);
          const int selectedActionIndex = dist3(eng);
          const auto &[eventCode1, observation1, actionIndex1] = eventObservationActionList[selectedActionIndex-1];
          const auto &[eventCode2, observation2, actionIndex2] = eventObservationActionList[selectedActionIndex];
          LOG(INFO) << "Selected actions " << selectedActionIndex-1 << " & " << selectedActionIndex << " from PVP #" << pvpId << " for observer " << worldState_.getEntity<entity::PlayerCharacter>(observerGlobalId)->name;
          jaxInterface_.train(observation1, actionIndex1, calculateReward(observation1, observation2), observation2);
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // jaxInterface_.train(previousObservation, action, reward, currentObservation);
  }
}

void TrainingManager::onUpdate(const event::Event *event) {
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

void TrainingManager::reportEventObservationAndAction(common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId observerGlobalId, const event::Event *event, const Observation &observation, int actionIndex) {
  // std::unordered_map<common::PvpDescriptor::PvpId, std::unordered_map<sro::scalar_types::EntityGlobalId, std::vector<std::tuple<event::EventCode, Observation, int>>>> newReplayBuffer_;
  // static std::unordered_map<common::PvpDescriptor::PvpId, std::unordered_map<sro::scalar_types::EntityGlobalId, std::vector<std::tuple<event::EventCode, Observation, int>>>> myMap;
  newReplayBuffer_[pvpId][observerGlobalId].push_back({event->eventCode, observation, actionIndex});

  static int replayCount=0;
  replayCount++;
  // if (replayCount % 100 == 0) {
  //   LOG(INFO) << "Replay count: " << replayCount;
  //   // Collect some stats about pvps in the replay buffer.
  //   for (const auto &[pvpId, observerMap] : newReplayBuffer_) {
  //     for (const auto &[observerGlobalId, eventObservationActionList] : observerMap) {
  //       LOG(INFO) << "[PVP #" << pvpId << "] " << worldState_.getEntity<entity::PlayerCharacter>(observerGlobalId)->name << " has " << eventObservationActionList.size() << " events";
  //       // Calculate episode return.
  //       double episodeReturn = 0.0;
  //       for (int i=1; i<eventObservationActionList.size(); ++i) {
  //         episodeReturn += calculateReward(std::get<1>(eventObservationActionList[i-1]), std::get<1>(eventObservationActionList[i]));
  //       }
  //       LOG(INFO) << "  Episode return: " << episodeReturn;
  //     }
  //   }
  // }
  // LOG(INFO) << "[PVP #" << pvpId << "] Given event " << event::toString(event->eventCode) << " and observation " << observation.toString() << " for observer " << observerGlobalId << " and action " << actionIndex;
}

// void TrainingManager::reportObservationAndAction(sro::scalar_types::EntityGlobalId observerGlobalId, const Observation &observation, int actionIndex) {
//   auto it = lastObservationMap_.find(observerGlobalId);
//   if (it != lastObservationMap_.end()) {
//     // We have a previous observation.
//     const LastObservationAndAction &lastObservationAndAction = it->second;
//     const double reward = calculateReward(lastObservationAndAction.observation, observation);
//     // We want to store S,A,R,S' in a replay buffer.
//     replayBuffer_.push_back({lastObservationAndAction.observation, lastObservationAndAction.actionIndex, reward, observation});
//     // LOG(INFO) << "Received observation " << observation.toString() << " and action " << actionIndex;
//     // LOG(INFO) << "Previous observation " << lastObservationAndAction.observation.toString() << ". Reward: " << reward << ", replay buffer size: " << replayBuffer_.size();
//     if (replayBuffer_.size() % 1000 == 0) {
//       LOG(INFO) << "Replay buffer size: " << replayBuffer_.size();
//     }
//   } else {
//     LOG(INFO) << "Received first observation";
//   }
//   lastObservationMap_[observerGlobalId] = LastObservationAndAction{observation, actionIndex};
// }

void TrainingManager::setUpIntelligencePool() {

}

void TrainingManager::createSessions() {
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

common::PvpDescriptor TrainingManager::buildPvpDescriptor(Session &char1, Session &char2) {
  common::PvpDescriptor pvpDescriptor;
  pvpDescriptor.pvpId = nextPvpId_++;
  pvpDescriptor.player1GlobalId = char1.getBot().selfState()->globalId;
  pvpDescriptor.player2GlobalId = char2.getBot().selfState()->globalId;

  // Massive Pvp Area
  const sro::Position pvpCenterPosition(/*regionId=*/sro::position_math::worldRegionIdFromSectors(1,1),
    /*xOffset=*/960.0,
    /*yOffset=*/ 20.0,
    /*zOffset=*/960.0);

  pvpDescriptor.pvpPositionPlayer1 = sro::position_math::createNewPositionWith2dOffset(pvpCenterPosition, +kPvpStartingCenterOffset, 0.0);
  pvpDescriptor.pvpPositionPlayer2 = sro::position_math::createNewPositionWith2dOffset(pvpCenterPosition, -kPvpStartingCenterOffset, 0.0);

  pvpDescriptor.itemRequirements = itemRequirements_;

  pvpDescriptor.player1Intelligence = intelligencePool_.getRandomIntelligence();
  pvpDescriptor.player2Intelligence = intelligencePool_.getDeepLearningIntelligence();

  return pvpDescriptor;
}

void TrainingManager::createAndPublishPvpDescriptor() {
  if (sessionsReadyForAssignment_.size() < 2) {
    throw std::runtime_error("Not enough sessions ready for assignment");
  }
  SessionId char1Id = sessionsReadyForAssignment_.at(0);
  SessionId char2Id = sessionsReadyForAssignment_.at(1);
  Session &char1 = getSession(char1Id);
  Session &char2 = getSession(char2Id);

  common::PvpDescriptor pvpDescriptor = buildPvpDescriptor(char1, char2);

  // ----------------------Send the PvpDescriptor----------------------
  eventBroker_.publishEvent<event::BeginPvp>(pvpDescriptor);

  // These two sessions should now no longer be considered for a new Pvp.
  sessionsReadyForAssignment_.erase(sessionsReadyForAssignment_.begin(), sessionsReadyForAssignment_.begin() + 2);
  LOG(INFO) << "Published BeginPvp event for " << char1Id << " and " << char2Id << ". Now have " << sessionsReadyForAssignment_.size() << " sessions ready for assignment";
}

Session& TrainingManager::getSession(SessionId sessionId) {
  for (const std::unique_ptr<Session> &sessionPtr : sessions_) {
    if (sessionPtr->sessionId() == sessionId) {
      return *sessionPtr.get();
    }
  }
  throw std::runtime_error("Session not found");
}

void TrainingManager::pvp(Bot &char1, Bot &char2) {
  // Start the fight by sending an event.
  eventBroker_.publishEvent(event::EventCode::kRlStartPvp);
}

void TrainingManager::buildItemRequirementList() {
  const sro::scalar_types::ReferenceObjectId smallHpPotionRefId = gameData_.itemData().getItemId([](const sro::pk2::ref::Item &item) {
    return type_id::categories::kHpPotion.contains(type_id::getTypeId(item)) && item.itemClass == 2;
  });
  const sro::scalar_types::ReferenceObjectId smallMpPotionRefId = gameData_.itemData().getItemId([](const sro::pk2::ref::Item &item) {
    return type_id::categories::kMpPotion.contains(type_id::getTypeId(item)) && item.itemClass == 2;
  });
  const sro::scalar_types::ReferenceObjectId mediumUniversalPillRefId = gameData_.itemData().getItemId([](const sro::pk2::ref::Item &item) {
    return type_id::categories::kUniversalPill.contains(type_id::getTypeId(item)) && item.itemClass == 2;
  });

  constexpr int kSmallHpPotionRequiredCount = 400;
  constexpr int kSmallMpPotionRequiredCount = 400;
  constexpr int kMediumUniversalPillRequiredCount = 200;
  itemRequirements_.push_back({smallHpPotionRefId, kSmallHpPotionRequiredCount});
  itemRequirements_.push_back({smallMpPotionRefId, kSmallMpPotionRequiredCount});
  itemRequirements_.push_back({mediumUniversalPillRefId, kMediumUniversalPillRequiredCount});
}

double TrainingManager::calculateReward(const Observation &lastObservation, const Observation &observation) const {
  double reward = 0.0;
  // We get some positive reward proportional to how much our health increased, negative if it decreased.
  reward += (static_cast<int64_t>(observation.ourCurrentHp_) - lastObservation.ourCurrentHp_) / static_cast<double>(observation.ourMaxHp_);
  // We get some positive reward proportional to how much our opponent's health decreased, negative if it increased.
  reward += (static_cast<int64_t>(lastObservation.opponentCurrentHp_) - observation.opponentCurrentHp_) / static_cast<double>(observation.opponentMaxHp_);
  return reward;
}

} // namespace rl