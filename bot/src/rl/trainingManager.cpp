#include "bot.hpp"
#include "characterLoginInfo.hpp"
#include "common/pvpDescriptor.hpp"
#include "rl/ai/randomIntelligence.hpp"
#include "rl/trainingManager.hpp"
#include "session.hpp"
#include "type_id/categories.hpp"
#include "ui/rlUserInterface.hpp"

#include "packet/building/clientAgentCharacterMoveRequest.hpp"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/position_math.hpp>

namespace rl {

TrainingManager::TrainingManager(const pk2::GameData &gameData,
                  broker::EventBroker &eventBroker,
                  ui::RlUserInterface &rlUserInterface,
                  state::WorldState &worldState,
                  ClientManagerInterface &clientManagerInterface) :
                      gameData_(gameData),
                      eventBroker_(eventBroker),
                      rlUserInterface_(rlUserInterface),
                      worldState_(worldState),
                      clientManagerInterface_(clientManagerInterface) {
  buildItemRequirementList();
}

void TrainingManager::run() {
  setUpIntelligencePool();

  auto eventHandleFunction = std::bind(&TrainingManager::onUpdate, this, std::placeholders::_1);
  // Subscribe to events.
  eventBroker_.subscribeToEvent(event::EventCode::kPvpManagerReadyForAssignment, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiStartTraining, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiStopTraining, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiRequestCheckpointList, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiSaveCheckpoint, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiLoadCheckpoint, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiDeleteCheckpoints, eventHandleFunction);

  jaxInterface_.initialize();

  // 1. Wait here until we're told to start training.
  // 2. Enter training loop once we're told to do so.
  // 3. Break loop if we're told to stop.
  // 4. Go to #1
  bool once = false;
  while (true) {
    LOG(INFO) << "Waiting to be told to train";
    std::unique_lock lock(runTrainingMutex_);
    runTrainingCondition_.wait(lock, [this]() -> bool { return runTraining_; });
    LOG(INFO) << "Starting training";
    if (once) {
      throw std::runtime_error("We do not yet support restarting training");
    }
    createSessions();
    train();
    once = true;
  }
}

void TrainingManager::train() {
  std::mt19937 randomEngine = common::createRandomEngine();
  while (runTraining_) {
    try {
      if (replayBuffer_.size() < replayBuffer_.samplingBatchSize()) {
        // We don't have enough transitions to sample from yet.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      // Sample a batch of transitions from the replay buffer.
      std::vector<ReplayBuffer::SampleResult> sampleResult = replayBuffer_.sample();

      std::vector<Observation> olderObservations;
      std::vector<int> actionIndexs;
      std::vector<bool> isTerminals;
      std::vector<float> rewards;
      std::vector<Observation> newerObservations;
      std::vector<float> weights;
      olderObservations.reserve(sampleResult.size());
      actionIndexs.reserve(sampleResult.size());
      isTerminals.reserve(sampleResult.size());
      rewards.reserve(sampleResult.size());
      newerObservations.reserve(sampleResult.size());
      weights.reserve(sampleResult.size());

      for (int i=0; i<sampleResult.size(); ++i) {
        const ReplayBuffer::SampleResult &sample = sampleResult.at(i);
        Observation observation0 = sample.transition.first.observation;
        const std::optional<int> actionIndex0 = sample.transition.first.actionIndex;
        const Observation observation1 = sample.transition.second.observation;
        const std::optional<int> actionIndex1 = sample.transition.second.actionIndex;
        float weight = sample.weight;

        const bool isTerminal = !actionIndex1.has_value();
        const float reward = calculateReward(observation0, observation1, isTerminal);

        olderObservations.push_back(observation0);
        actionIndexs.push_back(actionIndex0.value());
        isTerminals.push_back(isTerminal);
        rewards.push_back(reward);
        newerObservations.push_back(observation1);
        weights.push_back(weight);
      }

      const JaxInterface::TrainAuxOutput trainOutput = jaxInterface_.train(olderObservations, actionIndexs, isTerminals, rewards, newerObservations, weights);

      // Update the priorities of the sampled transitions in the replay buffer.
      std::vector<ReplayBuffer::StorageIndexType> storageIndices;
      std::vector<float> tdErrors;
      storageIndices.reserve(sampleResult.size());
      tdErrors.reserve(sampleResult.size());
      storageIndices.push_back(sampleResult.at(0).storageIndex);
      tdErrors.push_back(trainOutput.tdError);
      replayBuffer_.updatePriorities(storageIndices, tdErrors);

      jaxInterface_.addScalar("TD Error", trainOutput.tdError, trainStepCount_);
      jaxInterface_.addScalar("Min Q Value", trainOutput.minQValue, trainStepCount_);
      jaxInterface_.addScalar("Mean Q Value", trainOutput.meanQValue, trainStepCount_);
      jaxInterface_.addScalar("Max Q Value", trainOutput.maxQValue, trainStepCount_);
      ++trainStepCount_;
      if (trainStepCount_ % kTargetNetworkUpdateInterval == 0) {
        LOG(INFO) << "Train step #" << trainStepCount_ << ". Updating target network";
        jaxInterface_.updateTargetModel();
      }
    } catch (std::exception &ex) {
      LOG(ERROR) << "Caught exception while training: " << ex.what();
    }
  }
  LOG(INFO) << "Done with training loop. Exiting train()";
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
  } else if (event->eventCode == event::EventCode::kRlUiStartTraining) {
    // Start training.
    std::unique_lock lock(runTrainingMutex_);
    runTraining_ = true;
    runTrainingCondition_.notify_one();
  } else if (event->eventCode == event::EventCode::kRlUiStopTraining) {
    // Stop training.
    runTraining_ = false;
  } else if (event->eventCode == event::EventCode::kRlUiRequestCheckpointList) {
    const std::vector<std::string> checkpointNames = checkpointManager_.getCheckpointNames();
    rlUserInterface_.sendCheckpointList(checkpointNames);
  } else if (event->eventCode == event::EventCode::kRlUiSaveCheckpoint) {
    const auto *saveCheckpointEvent = dynamic_cast<const event::RlUiSaveCheckpoint*>(event);
    if (saveCheckpointEvent == nullptr) {
      throw std::runtime_error("Received kRlUiSaveCheckpoint event but failed to cast to event::RlUiSaveCheckpoint");
    }
    LOG(INFO) << "Received save checkpoint request for " << saveCheckpointEvent->checkpointName;
    saveCheckpoint(saveCheckpointEvent->checkpointName);
  } else if (event->eventCode == event::EventCode::kRlUiLoadCheckpoint) {
    const auto *loadCheckpointEvent = dynamic_cast<const event::RlUiLoadCheckpoint*>(event);
    if (loadCheckpointEvent == nullptr) {
      throw std::runtime_error("Received kRlUiLoadCheckpoint event but failed to cast to event::RlUiLoadCheckpoint");
    }
    LOG(INFO) << "Received load checkpoint request for " << loadCheckpointEvent->checkpointName;
    checkpointManager_.loadCheckpoint(loadCheckpointEvent->checkpointName, jaxInterface_, intelligencePool_.getDeepLearningIntelligence());
    LOG(INFO) << "Done loading";
  } else if (event->eventCode == event::EventCode::kRlUiDeleteCheckpoints) {
    const auto *deleteCheckpointsEvent = dynamic_cast<const event::RlUiDeleteCheckpoints*>(event);
    if (deleteCheckpointsEvent == nullptr) {
      throw std::runtime_error("Received kRlUiDeleteCheckpoints event but failed to cast to event::RlUiDeleteCheckpoints");
    }
    LOG(INFO) << "Received delete checkpoints request for " << deleteCheckpointsEvent->checkpointNames.size() << " checkpoints";
    checkpointManager_.deleteCheckpoints(deleteCheckpointsEvent->checkpointNames);
  } else {
    throw std::runtime_error("Received unknown event");
  }
}

void TrainingManager::reportObservationAndAction(common::PvpDescriptor::PvpId pvpId, const std::string &intelligenceName, const Observation &observation, std::optional<int> actionIndex) {
  ObservationAndActionStorage::Index index = replayBuffer_.addObservationAndAction(pvpId, intelligenceName, observation, actionIndex);
  ObservationAndActionStorage::ObservationAndActionType currentAction = replayBuffer_.getObservationAndAction(index);
  if (!actionIndex) {
    // This is the end of the episode, calculate & report the episode return.
    float cumulativeReward = 0.0f;
    // Go backwards and sum this agent's rewards so far.
    while (index.hasPrevious()) {
      const ObservationAndActionStorage::Index previousIndex = index.previous();
      const ObservationAndActionStorage::ObservationAndActionType previousAction = replayBuffer_.getObservationAndAction(previousIndex);
      const bool isTerminal = !currentAction.actionIndex.has_value();
      cumulativeReward += calculateReward(previousAction.observation, currentAction.observation, isTerminal);
      index = previousIndex;
      currentAction = previousAction;
    }
    jaxInterface_.addScalar(absl::StrFormat("Episode Return %s", intelligenceName), cumulativeReward, trainStepCount_);
  }
}

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
  if (!char1.getBot().selfState()) {
    throw std::runtime_error("char1 does not have a self state");
  }
  if (!char2.getBot().selfState()) {
    throw std::runtime_error("char2 does not have a self state");
  }
  pvpDescriptor.player1Name = char1.getBot().selfState()->name;
  pvpDescriptor.player2Name = char2.getBot().selfState()->name;

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
  LOG(INFO) << "Published BeginPvp event for " << char1.getBot().selfState()->name << " & " << char2.getBot().selfState()->name << ". Now have " << sessionsReadyForAssignment_.size() << " sessions ready for assignment";
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

  constexpr int kSmallHpPotionRequiredCount = 5; // IF-CHANGE: If we change this, also change the max potion count in JaxInterface::observationToNumpy
  constexpr int kSmallMpPotionRequiredCount = 5;
  constexpr int kMediumUniversalPillRequiredCount = 5;
  itemRequirements_.push_back({smallHpPotionRefId, kSmallHpPotionRequiredCount});
  itemRequirements_.push_back({smallMpPotionRefId, kSmallMpPotionRequiredCount});
  itemRequirements_.push_back({mediumUniversalPillRefId, kMediumUniversalPillRequiredCount});
}

float TrainingManager::calculateReward(const Observation &lastObservation, const Observation &observation, bool isTerminal) const {
  float reward = 0.0f;
  // We get some positive reward proportional to how much our health increased, negative if it decreased.
  reward += (static_cast<int64_t>(observation.ourCurrentHp_) - lastObservation.ourCurrentHp_) / static_cast<double>(observation.ourMaxHp_);
  // We get some positive reward proportional to how much our opponent's health decreased, negative if it increased.
  reward += (static_cast<int64_t>(lastObservation.opponentCurrentHp_) - observation.opponentCurrentHp_) / static_cast<double>(observation.opponentMaxHp_);
  if (isTerminal) {
    // Give an extra bump for a win or loss.
    if (observation.ourCurrentHp_ == 0) {
      reward -= 2.0f;
    } else if (observation.opponentCurrentHp_ == 0) {
      reward += 2.0f;
    }
  }
  return reward;
}

void TrainingManager::saveCheckpoint(const std::string &checkpointName) {
  LOG(INFO) << "Being asked to save checkpoint \"" << checkpointName << "\"";
  const bool checkpointAlreadyExists = checkpointManager_.checkpointExists(checkpointName);
  if (checkpointAlreadyExists) {
    LOG(INFO) << "  Checkpoint already exists";
    rlUserInterface_.sendCheckpointAlreadyExists(checkpointName);
    return;
  }
  LOG(INFO) << "  Checkpoint does not yet exist";

  // Checkpoint does not exist. Save it.
  // What needs to be saved in the checkpoint?
  //  - Current model weights
  //  - Target model weights
  //  - Optimizer state
  //  - DeepLearningIntelligence::stepCount_
  //  - Replay buffer, maybe?
  // We'll let the
  checkpointManager_.saveCheckpoint(checkpointName, jaxInterface_, intelligencePool_.getDeepLearningIntelligence()->getStepCount());
}

} // namespace rl