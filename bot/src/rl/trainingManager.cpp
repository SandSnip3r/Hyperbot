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

  jaxInterface_.initialize(kObservationStackSize);

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
      if (replayBuffer_.size() < kBatchSize) {
        // We don't have enough transitions to sample from yet.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      const float beta = std::min(kPerBetaEnd, kPerBetaStart + (kPerBetaEnd - kPerBetaStart) * (static_cast<float>(trainStepCount_) / static_cast<float>(kPerTrainStepCountAnneal)));
      jaxInterface_.addScalar("anneal/Beta", beta, trainStepCount_);

      // Sample a batch of transitions from the replay buffer.
      std::vector<ReplayBufferType::SampleResult> sampleResult = replayBuffer_.sample(kBatchSize, randomEngine, beta);

      // Track training rate
      const std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
      const std::chrono::high_resolution_clock::duration timeDiff = currentTime - lastTrainingTime_;
      if (timeDiff > kTrainRateReportInterval) {
        const float trainingRate = static_cast<float>(trainingCount_) / (std::chrono::duration_cast<std::chrono::milliseconds>(timeDiff).count()/1000.0);
        jaxInterface_.addScalar("perf/Training Rate", trainingRate, trainStepCount_);
        trainingCount_ = 0;
        lastTrainingTime_ = currentTime;
      }
      trainingCount_ += kBatchSize;

      const ModelInputs modelInputs = buildModelInputsFromReplayBufferSamples(sampleResult);

      const JaxInterface::TrainAuxOutput trainOutput = jaxInterface_.train(kObservationStackSize,
                                                                           modelInputs.olderObservationStacks,
                                                                           modelInputs.actionIndexs,
                                                                           modelInputs.isTerminals,
                                                                           modelInputs.rewards,
                                                                           modelInputs.newerObservationStacks,
                                                                           modelInputs.weights);

      // Update the priorities of the sampled transitions in the replay buffer.
      std::vector<ReplayBufferType::TransitionId> ids;
      std::vector<float> newPriorities;
      ids.reserve(sampleResult.size());
      newPriorities.reserve(sampleResult.size());

      if (sampleResult.size() != trainOutput.tdErrors.size()) {
        throw std::runtime_error(absl::StrFormat("Sample size %zu does not match TD error size %zu", sampleResult.size(), trainOutput.tdErrors.size()));
      }
      for (int i=0; i<sampleResult.size(); ++i) {
        ids.push_back(sampleResult.at(i).transitionId);
        newPriorities.push_back(std::abs(trainOutput.tdErrors.at(i)));
      }
      replayBuffer_.updatePriorities(ids, newPriorities);

      const float meanTdError = std::accumulate(trainOutput.tdErrors.begin(), trainOutput.tdErrors.end(), 0.0f) / trainOutput.tdErrors.size();
      jaxInterface_.addScalar("TD Error", meanTdError, trainStepCount_);
      jaxInterface_.addScalar("Q_Value/Min", trainOutput.meanMinQValue, trainStepCount_);
      jaxInterface_.addScalar("Q_Value/Mean", trainOutput.meanMeanQValue, trainStepCount_);
      jaxInterface_.addScalar("Q_Value/Max", trainOutput.meanMaxQValue, trainStepCount_);
      ++trainStepCount_;
      if (kUsePolyakTargetNetworkUpdate) {
        // Soft update target network every step with Polyak averaging
        jaxInterface_.updateTargetModelPolyak(kTargetNetworkPolyakTau);
      } else if (trainStepCount_ % kTargetNetworkUpdateInterval == 0) {
        // Hard update target network at fixed intervals
        LOG(INFO) << "Train step #" << trainStepCount_ << ". Updating target network";
        jaxInterface_.updateTargetModel();
      }
      if (trainStepCount_ % kTrainStepCheckpointInterval == 0) {
        LOG(INFO) << "Train step #" << trainStepCount_ << ". Saving checkpoint";
        saveCheckpoint("backup_checkpoint", /*overwrite=*/true);
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
    saveCheckpoint(saveCheckpointEvent->checkpointName, /*overwrite=*/false);
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
  using Id = ObservationAndActionStorage::Id;
  using ObservationAndActionType = ObservationAndActionStorage::ObservationAndActionType;

  // Store this item in the observation and action storage.
  std::pair<Id, std::vector<Id>> observationIdAndDeletedObservationIds = observationAndActionStorage_.addObservationAndAction(pvpId, intelligenceName, observation, actionIndex);

  const std::vector<Id> &deletedObservationIds = observationIdAndDeletedObservationIds.second;
  if (deletedObservationIds.size() > 0) {
    // Some observations were deleted due to the buffer being full. Remove the corresponding transitions from the replay buffer.
    for (Id deletedObservationId : deletedObservationIds) {
      std::unique_lock lock(observationTransitionIdMapMutex_);
      auto it = observationIdToTransitionIdMap_.find(deletedObservationId);
      if (it == observationIdToTransitionIdMap_.end()) {
        // No transition ID for this observation ID. This can happen if the observation was the first in a new PVP.
        // It will always be the case that one deleted observation will not have a corresponding transition ID. That is because we only have transition IDs for pairs of observations.
        continue;
      }
      const ReplayBufferType::TransitionId transitionId = it->second;
      replayBuffer_.deleteTransition(transitionId);
      auto it2 = transitionIdToObservationIdMap_.find(transitionId);
      const ObservationAndActionStorage::Id observationId = it2->second;
      observationIdToTransitionIdMap_.erase(observationId);
      transitionIdToObservationIdMap_.erase(transitionId);
    }
  }

  const Id observationId = observationIdAndDeletedObservationIds.first;
  if (observationAndActionStorage_.hasPrevious(observationId)) {
    // There was an observation before this one. We can store a full transition in the replay buffer (a transition traditionally is a (S,A,R,S') tuple).
    const ReplayBufferType::TransitionId transitionId = replayBuffer_.addTransition(observationId);
    std::unique_lock lock(observationTransitionIdMapMutex_);
    observationIdToTransitionIdMap_[observationId] = transitionId;
    transitionIdToObservationIdMap_[transitionId] = observationId;

    // Track sample collection rate
    sampleCount_++;
    const std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
    const std::chrono::high_resolution_clock::duration sampleCountTimeDiff = currentTime - lastSampleTime_;
    if (sampleCountTimeDiff > kSampleRateReportInterval) {
      const float sampleRate = static_cast<float>(sampleCount_) / (std::chrono::duration_cast<std::chrono::milliseconds>(sampleCountTimeDiff).count()/1000.0);
      jaxInterface_.addScalar("perf/Sample Collection Rate", sampleRate, trainStepCount_);
      sampleCount_ = 0;
      lastSampleTime_ = currentTime;
    }

    // Track replay buffer size
    const std::chrono::high_resolution_clock::duration replayBufferSizeTimeDiff = currentTime - lastReplayBufferSizeUpdateTime_;
    if (replayBufferSizeTimeDiff > kReplayBufferSizeUpdateInterval) {
      const int replayBufferSize = replayBuffer_.size();
      jaxInterface_.addScalar("perf/Replay Buffer Size", replayBufferSize, trainStepCount_);
      lastReplayBufferSizeUpdateTime_ = currentTime;
    }
  }

  if (!actionIndex) {
    // This is the end of the episode, calculate & report the episode return.
    float cumulativeReward = 0.0f;
    Id currentObservationId = observationId;
    ObservationAndActionType currentObservationAndAction = observationAndActionStorage_.getObservationAndAction(currentObservationId);
    // Go backwards and sum this agent's rewards.
    while (observationAndActionStorage_.hasPrevious(currentObservationId)) {
      Id previousObservationId = observationAndActionStorage_.getPreviousId(currentObservationId);
      ObservationAndActionType previousObservationAndAction = observationAndActionStorage_.getObservationAndAction(previousObservationId);
      const bool isTerminal = !currentObservationAndAction.actionIndex.has_value();
      const float reward = calculateReward(previousObservationAndAction.observation, currentObservationAndAction.observation, isTerminal);
      cumulativeReward += reward;
      currentObservationId = previousObservationId;
      currentObservationAndAction = previousObservationAndAction;
    }
    jaxInterface_.addScalar(absl::StrFormat("Episode_Return/%s", intelligenceName), cumulativeReward, trainStepCount_);
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

  pvpDescriptor.player1Intelligence = intelligencePool_.getDeepLearningIntelligence();
  pvpDescriptor.player2Intelligence = intelligencePool_.getRandomIntelligence();
  pvpDescriptor.player1Intelligence->resetForNewEpisode();
  pvpDescriptor.player2Intelligence->resetForNewEpisode();

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
  eventBroker_.publishEvent<event::BeginPvp>(std::move(pvpDescriptor));

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
  // We get some negative reward if the event is an error.
  if (observation.eventCode_ == event::EventCode::kCommandError ||
      observation.eventCode_ == event::EventCode::kItemUseFailed) {
    reward -= 0.0005f;
  }
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

void TrainingManager::saveCheckpoint(const std::string &checkpointName, bool overwrite) {
  LOG(INFO) << "Being asked to save checkpoint \"" << checkpointName << "\"";
  if (!overwrite) {
    const bool checkpointAlreadyExists = checkpointManager_.checkpointExists(checkpointName);
    if (checkpointAlreadyExists) {
      LOG(INFO) << "  Checkpoint already exists";
      rlUserInterface_.sendCheckpointAlreadyExists(checkpointName);
      return;
    }
    LOG(INFO) << "  Checkpoint does not yet exist";
  }

  checkpointManager_.saveCheckpoint(checkpointName, jaxInterface_, intelligencePool_.getDeepLearningIntelligence()->getStepCount(), overwrite);
}

TrainingManager::ModelInputs TrainingManager::buildModelInputsFromReplayBufferSamples(const std::vector<ReplayBufferType::SampleResult> &samples) const {
  ZoneScopedN("TrainingManager::buildModelInputsFromReplayBufferSamples");
  ModelInputs modelInputs;
  const size_t sampleSize = samples.size();

  // Pre-allocate all vectors to exact size.
  modelInputs.olderObservationStacks.reserve(sampleSize);
  modelInputs.actionIndexs.reserve(sampleSize);
  modelInputs.isTerminals.reserve(sampleSize);
  modelInputs.rewards.reserve(sampleSize);
  modelInputs.newerObservationStacks.reserve(sampleSize);
  modelInputs.weights.reserve(sampleSize);

  // Batch look up all observation IDs at once to minimize mutex lock time.
  std::vector<ObservationAndActionStorage::Id> observationIds(sampleSize);
  {
    std::unique_lock lock(observationTransitionIdMapMutex_);
    for (int i=0; i<sampleSize; ++i) {
      observationIds[i] = transitionIdToObservationIdMap_.at(samples[i].transitionId);
    }
  }

  for (int i=0; i<samples.size(); ++i) {
    const ReplayBufferType::SampleResult &sample = samples.at(i);
    const ObservationAndActionStorage::Id observationId = observationIds.at(i);
    const ObservationAndActionStorage::Id previousObservationId = observationAndActionStorage_.getPreviousId(observationId);
    const ObservationAndActionStorage::ObservationAndActionType &previousObservationAndAction = observationAndActionStorage_.getObservationAndAction(previousObservationId);
    const Observation &observation0 = previousObservationAndAction.observation;
    const std::optional<int> &actionIndex0 = previousObservationAndAction.actionIndex;

    const ObservationAndActionStorage::ObservationAndActionType &observationAndAction = observationAndActionStorage_.getObservationAndAction(observationId);
    const Observation &observation1 = observationAndAction.observation;
    const std::optional<int> &actionIndex1 = observationAndAction.actionIndex;

    const float weight = sample.weight;

    const bool isTerminal = !actionIndex1.has_value();
    const float reward = calculateReward(observation0, observation1, isTerminal);

    std::vector<Observation> olderObservationStack;
    olderObservationStack.reserve(kObservationStackSize);
    olderObservationStack.push_back(observation0);
    {
      ObservationAndActionStorage::Id tmp = previousObservationId;
      while (olderObservationStack.size() < kObservationStackSize && observationAndActionStorage_.hasPrevious(tmp)) {
        tmp = observationAndActionStorage_.getPreviousId(tmp);
        const ObservationAndActionStorage::ObservationAndActionType &observationAndAction = observationAndActionStorage_.getObservationAndAction(tmp);
        olderObservationStack.push_back(observationAndAction.observation);
      }
      std::reverse(olderObservationStack.begin(), olderObservationStack.end());
    }
    modelInputs.olderObservationStacks.push_back(std::move(olderObservationStack));

    modelInputs.actionIndexs.push_back(actionIndex0.value());
    modelInputs.isTerminals.push_back(isTerminal);
    modelInputs.rewards.push_back(reward);

    std::vector<Observation> newerObservationStack;
    newerObservationStack.reserve(kObservationStackSize);
    newerObservationStack.push_back(observation1);
    {
      ObservationAndActionStorage::Id tmp = observationId;
      while (newerObservationStack.size() < kObservationStackSize && observationAndActionStorage_.hasPrevious(tmp)) {
        tmp = observationAndActionStorage_.getPreviousId(tmp);
        const ObservationAndActionStorage::ObservationAndActionType &observationAndAction = observationAndActionStorage_.getObservationAndAction(tmp);
        newerObservationStack.push_back(observationAndAction.observation);
      }
      std::reverse(newerObservationStack.begin(), newerObservationStack.end());
    }
    modelInputs.newerObservationStacks.push_back(std::move(newerObservationStack));

    modelInputs.weights.push_back(weight);
  }

  return modelInputs;
}

} // namespace rl