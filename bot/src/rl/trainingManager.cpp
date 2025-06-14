#include "bot.hpp"
#include "characterLoginInfo.hpp"
#include "common/pvpDescriptor.hpp"
#include "rl/ai/deepLearningIntelligence.hpp"
#include "rl/ai/randomIntelligence.hpp"
#include "rl/trainingManager.hpp"
#include "session.hpp"
#include "type_id/categories.hpp"
#include "ui/rlUserInterface.hpp"

#include "packet/building/clientAgentCharacterMoveRequest.hpp"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/position_math.hpp>

#include <absl/strings/str_join.h>
namespace rl {

TrainingManager::TrainingManager(const sro::pk2::GameData &gameData,
                  broker::EventBroker &eventBroker,
                  ui::RlUserInterface &rlUserInterface,
                  state::WorldState &worldState,
                  ClientManagerInterface &clientManagerInterface) :
                      gameData_(gameData),
                      eventBroker_(eventBroker),
                      rlUserInterface_(rlUserInterface),
                      worldState_(worldState),
                      clientManagerInterface_(clientManagerInterface) {
}

void TrainingManager::run() {
  buildItemRequirementList();
  defineCharacterPairingsAndPositions();

  auto eventHandleFunction = std::bind(&TrainingManager::onUpdate, this, std::placeholders::_1);
  // Subscribe to events.
  eventBroker_.subscribeToEvent(event::EventCode::kPvpManagerReadyForAssignment, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiStartTraining, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiStopTraining, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiRequestCheckpointList, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiSaveCheckpoint, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiLoadCheckpoint, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRlUiDeleteCheckpoints, eventHandleFunction);

  jaxInterface_.initialize(kDropoutRate);
  precompileModels();

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
  JaxInterface::Model model = jaxInterface_.getModel();
  JaxInterface::Optimizer optimizer = jaxInterface_.getOptimizer();
  JaxInterface::Model targetModel = jaxInterface_.getTargetModel();
  while (runTraining_) {
    try {
      std::unique_lock lock(replayBufferAndStorageMutex_);
      if (replayBuffer_.size() < kBatchSize || replayBuffer_.size() < kReplayBufferMinimumBeforeTraining) {
        // We don't have enough transitions to sample from yet.
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      // Sample a batch of transitions from the replay buffer.
      const float beta = std::min(kPerBetaEnd, kPerBetaStart + (kPerBetaEnd - kPerBetaStart) * (static_cast<float>(trainStepCount_) / static_cast<float>(kPerTrainStepCountAnneal)));
      const std::vector<ReplayBufferType::SampleResult> sampleResult = replayBuffer_.sample(kBatchSize, randomEngine, beta);
      holdingSample_ = true;

      const model_inputs::BatchedTrainingInput modelInputs = buildModelInputsFromReplayBufferSamples(sampleResult);
      lock.unlock();

      jaxInterface_.addScalar("anneal/Beta", beta, trainStepCount_);

      // Track training rate
      const std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
      const std::chrono::steady_clock::duration timeDiff = currentTime - lastTrainingTime_;
      if (timeDiff > kTrainRateReportInterval) {
        const float trainingRate = static_cast<float>(trainingCount_) / (std::chrono::duration_cast<std::chrono::milliseconds>(timeDiff).count()/1000.0);
        jaxInterface_.addScalar("perf/Training Rate", trainingRate, trainStepCount_);
        trainingCount_ = 0;
        lastTrainingTime_ = currentTime;
      }
      trainingCount_ += kBatchSize;


      const JaxInterface::TrainAuxOutput trainOutput = jaxInterface_.train(model,
                                                                           optimizer,
                                                                           targetModel,
                                                                           modelInputs.oldModelInputViews,
                                                                           modelInputs.actionsTaken,
                                                                           modelInputs.isTerminals,
                                                                           modelInputs.rewards,
                                                                           modelInputs.newModelInputViews,
                                                                           modelInputs.importanceSamplingWeights);

      // Update the priorities of the sampled transitions in the replay buffer.
      std::vector<ReplayBufferType::TransitionId> ids;
      std::vector<float> newPriorities;
      ids.reserve(sampleResult.size());
      newPriorities.reserve(sampleResult.size());

      {
        std::unique_lock lock(replayBufferAndStorageMutex_);
        for (int i=0; i<sampleResult.size(); ++i) {
          if (deletedTransitionIds_.find(sampleResult.at(i).transitionId) != deletedTransitionIds_.end()) {
            // This transition was deleted, skip it.
            continue;
          }
          ids.push_back(sampleResult.at(i).transitionId);
          newPriorities.push_back(std::abs(trainOutput.tdErrors.at(i)));
        }
        replayBuffer_.updatePriorities(ids, newPriorities);
        deletedTransitionIds_.clear();
        holdingSample_ = false;
      }

      const float minTdError = *std::min_element(trainOutput.tdErrors.begin(), trainOutput.tdErrors.end());
      const float meanTdError = std::accumulate(trainOutput.tdErrors.begin(), trainOutput.tdErrors.end(), 0.0f) / trainOutput.tdErrors.size();
      const float maxTdError = *std::max_element(trainOutput.tdErrors.begin(), trainOutput.tdErrors.end());
      jaxInterface_.addScalar("Global Norm", trainOutput.globalNorm, trainStepCount_);
      jaxInterface_.addScalar("TD_Error/Min", minTdError, trainStepCount_);
      jaxInterface_.addScalar("TD_Error/Mean", meanTdError, trainStepCount_);
      jaxInterface_.addScalar("TD_Error/Max", maxTdError, trainStepCount_);
      jaxInterface_.addScalar("Q_Value/Min", trainOutput.meanMinQValue, trainStepCount_);
      jaxInterface_.addScalar("Q_Value/Mean", trainOutput.meanMeanQValue, trainStepCount_);
      jaxInterface_.addScalar("Q_Value/Max", trainOutput.meanMaxQValue, trainStepCount_);
      ++trainStepCount_;
      if (kTargetNetworkPolyakEnabled) {
        if (trainStepCount_ % kTargetNetworkPolyakUpdateInterval == 0) {
          // Soft update target network every step with Polyak averaging
          jaxInterface_.updateTargetModelPolyak(kTargetNetworkPolyakTau);
        }
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

    // Check if any character pairings can now be assigned
    checkAndPublishPvpDescriptors();
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
    const CheckpointValues checkpointValues = checkpointManager_.loadCheckpoint(loadCheckpointEvent->checkpointName, jaxInterface_);
    trainStepCount_ = checkpointValues.stepCount;
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
  std::optional<float> sampleRate;
  std::optional<int> replayBufferSize;
  std::optional<float> cumulativeReward;
  {
    std::unique_lock lock(replayBufferAndStorageMutex_);
    using Id = ObservationAndActionStorage::Id;
    using ObservationAndActionType = ObservationAndActionStorage::ObservationAndActionType;

    // Store this item in the observation and action storage.
    const std::pair<Id, std::vector<Id>> observationIdAndDeletedObservationIds = observationAndActionStorage_.addObservationAndAction(pvpId, intelligenceName, observation, actionIndex);

    const std::vector<Id> &deletedObservationIds = observationIdAndDeletedObservationIds.second;
    if (deletedObservationIds.size() > 0) {
      // Some observations were deleted due to the buffer being full. Remove the corresponding transitions from the replay buffer.
      for (Id deletedObservationId : deletedObservationIds) {
        auto it = observationIdToTransitionIdMap_.find(deletedObservationId);
        if (it == observationIdToTransitionIdMap_.end()) {
          // No transition ID for this observation ID. This can happen if the observation was the first in a new PVP.
          // It will always be the case that one deleted observation will not have a corresponding transition ID. That is because we only have transition IDs for pairs of observations.
          continue;
        }
        const ReplayBufferType::TransitionId transitionId = it->second;
        if (holdingSample_) {
          // In the training thread, we are currently working with a sample from the replay buffer. Since we're deleting some now, we need to make sure in the training thread to not try to update the IDs of those items.
          deletedTransitionIds_.insert(transitionId);
        }
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
      observationIdToTransitionIdMap_[observationId] = transitionId;
      transitionIdToObservationIdMap_[transitionId] = observationId;

      // Track sample collection rate
      sampleCount_++;
      const std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
      const std::chrono::steady_clock::duration sampleCountTimeDiff = currentTime - lastSampleTime_;
      if (sampleCountTimeDiff > kSampleRateReportInterval) {
        sampleRate = static_cast<float>(sampleCount_) / (std::chrono::duration_cast<std::chrono::milliseconds>(sampleCountTimeDiff).count()/1000.0);
        sampleCount_ = 0;
        lastSampleTime_ = currentTime;
      }

      // Track replay buffer size
      const std::chrono::steady_clock::duration replayBufferSizeTimeDiff = currentTime - lastReplayBufferSizeUpdateTime_;
      if (replayBufferSizeTimeDiff > kReplayBufferSizeUpdateInterval) {
        replayBufferSize = replayBuffer_.size();
        lastReplayBufferSizeUpdateTime_ = currentTime;
      }
    }

    if (!actionIndex) {
      // This is the end of the episode, calculate & report the episode return.
      cumulativeReward = 0.0f;
      Id currentObservationId = observationId;
      const ObservationAndActionType *currentObservationAndAction = &observationAndActionStorage_.getObservationAndAction(currentObservationId);
      // Go backwards and sum this agent's rewards.
      while (observationAndActionStorage_.hasPrevious(currentObservationId)) {
        const Id previousObservationId = observationAndActionStorage_.getPreviousId(currentObservationId);
        const ObservationAndActionType &previousObservationAndAction = observationAndActionStorage_.getObservationAndAction(previousObservationId);
        const bool isTerminal = !currentObservationAndAction->actionIndex.has_value();
        const float reward = calculateReward(previousObservationAndAction.observation, currentObservationAndAction->observation, isTerminal);
        *cumulativeReward += reward;
        currentObservationId = previousObservationId;
        currentObservationAndAction = &previousObservationAndAction;
      }
    }
  }

  // Report metrics.
  jaxInterface_.addScalar("anneal/Epsilon", getEpsilon(), trainStepCount_);
  if (sampleRate) {
    jaxInterface_.addScalar("perf/Sample Collection Rate", *sampleRate, trainStepCount_);
  }
  if (replayBufferSize) {
    jaxInterface_.addScalar("perf/Replay Buffer Size", *replayBufferSize, trainStepCount_);
  }
  if (cumulativeReward) {
    jaxInterface_.addScalar(absl::StrFormat("Episode_Return/%s", intelligenceName), *cumulativeReward, trainStepCount_);
  }
}

float TrainingManager::getEpsilon() const {
  return std::min(kInitialEpsilon, std::max(kFinalEpsilon, kInitialEpsilon - static_cast<float>(trainStepCount_) / kEpsilonDecaySteps));
}

void TrainingManager::createSessions() {
  LOG(INFO) << "Creating sessions for " << characterPairings_.size() << " total character pairings";

  static constexpr bool kClientless = true;
  static constexpr int kNumCharacterPairingClientsToStartAtATime = 4;

  size_t characterPairingIndex = 0;
  while (characterPairingIndex < characterPairings_.size()) {
    std::vector<Session*> sessions;
    std::vector<std::future<void>> clientOpenFutures;
    for (size_t i=characterPairingIndex; (kClientless || i<characterPairingIndex+kNumCharacterPairingClientsToStartAtATime) && i < characterPairings_.size(); ++i) {
      const CharacterPairing &pairing = characterPairings_.at(i);
      sessions_.push_back(std::make_unique<Session>(gameData_, eventBroker_, worldState_, clientManagerInterface_, rlUserInterface_));
      Session& session1 = *sessions_.back().get();
      sessions_.push_back(std::make_unique<Session>(gameData_, eventBroker_, worldState_, clientManagerInterface_, rlUserInterface_));
      Session& session2 = *sessions_.back().get();

      // Initialize sessions
      session1.initialize();
      session2.initialize();

      // Set characters
      session1.setCharacter(pairing.character1);
      session2.setCharacter(pairing.character2);

      // Start the sessions
      session1.runAsync();
      session2.runAsync();

      // Open clients
      sessions.push_back(&session1);
      sessions.push_back(&session2);

      if constexpr (kClientless) {
        clientOpenFutures.push_back(session1.connectClientlessAsync());
        clientOpenFutures.push_back(session2.connectClientlessAsync());
      } else {
        clientOpenFutures.push_back(session1.asyncOpenClient());
        clientOpenFutures.push_back(session2.asyncOpenClient());
      }
    }

    // Wait on all clients to open
    LOG(INFO) << "Waiting for " << clientOpenFutures.size() << " clients to open";
    for (std::future<void> &clientOpenFuture : clientOpenFutures) {
      clientOpenFuture.wait();
    }
    clientOpenFutures.clear();

    // Log in all sessions
    LOG(INFO) << "Clients are open. Logging in characters";
    std::vector<std::future<void>> botLoginFutures;
    for (Session* session : sessions) {
      // Log in bot
      Bot& bot = session->getBot();
      botLoginFutures.push_back(bot.asyncLogIn());
    }

    // The above calls have constructed login state machines within each bot.
    // Unfortunately, if there are no events on the event bus, the state machines will not progress.
    // We'll send one dummy event to kick them off.
    eventBroker_.publishEvent(event::EventCode::kDummy);

    // Wait for all characters to finish logging in
    LOG(INFO) << "Waiting for " << botLoginFutures.size() << " characters to log in";
    for (std::future<void> &botLoginFuture : botLoginFutures) {
      botLoginFuture.wait();
    }
    LOG(INFO) << "These " << botLoginFutures.size() << " characters are logged in.";
    botLoginFutures.clear();

    // Save session IDs in the pairings
    for (size_t i=characterPairingIndex; (kClientless || i<characterPairingIndex+kNumCharacterPairingClientsToStartAtATime) && i < characterPairings_.size(); ++i) {
      CharacterPairing &pairing = characterPairings_.at(i);
      Session& session1 = *sessions.at((i-characterPairingIndex)*2);
      Session& session2 = *sessions.at((i-characterPairingIndex)*2 + 1);
      pairing.session1Id = session1.sessionId();
      pairing.session2Id = session2.sessionId();
    }

    if constexpr (!kClientless) {
      characterPairingIndex += kNumCharacterPairingClientsToStartAtATime;
    } else {
      // In clientless mode, we can start all character pairings at once.
      characterPairingIndex = characterPairings_.size();
    }
  }

  // Standby for PVP
  LOG(INFO) << "Telling all characters to prepare for PVP";
  for (std::unique_ptr<Session> &session : sessions_) {
    Bot& bot = session->getBot();
    bot.asyncStandbyForPvp();
  }
  // Similar to the bot login state machines, asking bots to standby constructs a state machine for each bot.
  // We will again send a dummy event to ensure that they have a chance to progress.
  eventBroker_.publishEvent(event::EventCode::kDummy);

  LOG(INFO) << "All sessions created and ready for PVP. Total active sessions: " << sessions_.size();
}

common::PvpDescriptor TrainingManager::buildPvpDescriptor(Session &char1, Session &char2, int positionIndex) {
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

  // Use the specified position from our position list
  if (positionIndex < 0 || positionIndex >= pvpPositions_.size()) {
    throw std::runtime_error("Invalid position index: " + std::to_string(positionIndex));
  }

  const sro::Position& pvpCenterPosition = pvpPositions_[positionIndex];

  pvpDescriptor.pvpPositionPlayer1 = sro::position_math::createNewPositionWith2dOffset(pvpCenterPosition, +kPvpStartingCenterOffset, 0.0);
  pvpDescriptor.pvpPositionPlayer2 = sro::position_math::createNewPositionWith2dOffset(pvpCenterPosition, -kPvpStartingCenterOffset, 0.0);

  pvpDescriptor.itemRequirements = itemRequirements_;

  pvpDescriptor.player1Intelligence = std::make_shared<rl::ai::RandomIntelligence>(*this);
  pvpDescriptor.player2Intelligence = std::make_shared<rl::ai::DeepLearningIntelligence>(*this);

  return pvpDescriptor;
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

  constexpr int kSmallHpPotionRequiredCount = 5; // IF-CHANGE: If we change this, also change the max potion count in JaxInterface::writeObservationToRawArray
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

  checkpointManager_.saveCheckpoint(checkpointName, jaxInterface_, trainStepCount_, overwrite);
}

model_inputs::BatchedTrainingInput TrainingManager::buildModelInputsFromReplayBufferSamples(const std::vector<ReplayBufferType::SampleResult> &samples) const {
  ZoneScopedN("TrainingManager::buildModelInputsFromReplayBufferSamples");
  model_inputs::BatchedTrainingInput batchedTrainingInput;
  const size_t sampleSize = samples.size();

  // Pre-allocate all vectors to exact size.
  batchedTrainingInput.oldModelInputViews.reserve(sampleSize);
  batchedTrainingInput.actionsTaken.reserve(sampleSize);
  batchedTrainingInput.isTerminals.reserve(sampleSize);
  batchedTrainingInput.rewards.reserve(sampleSize);
  batchedTrainingInput.newModelInputViews.reserve(sampleSize);
  batchedTrainingInput.importanceSamplingWeights.reserve(sampleSize);

  // Batch look up all observation IDs at once.
  std::vector<ObservationAndActionStorage::Id> observationIds(sampleSize);
  for (int i=0; i<sampleSize; ++i) {
    observationIds[i] = transitionIdToObservationIdMap_.at(samples[i].transitionId);
  }

  for (int i=0; i<samples.size(); ++i) {
    const ReplayBufferType::SampleResult &sample = samples.at(i);

    const ObservationAndActionStorage::Id observationId = observationIds.at(i);
    const ObservationAndActionStorage::Id previousObservationId = observationAndActionStorage_.getPreviousId(observationId);

    const ObservationAndActionStorage::ObservationAndActionType &previousObservationAndAction = batchedTrainingInput.getObservationAndAction(previousObservationId, observationAndActionStorage_);
    const Observation &observation0 = previousObservationAndAction.observation;
    const std::optional<int> &actionIndex0 = previousObservationAndAction.actionIndex;

    const ObservationAndActionStorage::ObservationAndActionType &observationAndAction = batchedTrainingInput.getObservationAndAction(observationId, observationAndActionStorage_);
    const Observation &observation1 = observationAndAction.observation;
    const std::optional<int> &actionIndex1 = observationAndAction.actionIndex;

    const bool isTerminal = !actionIndex1.has_value();
    const float reward = calculateReward(observation0, observation1, isTerminal);
    const float weight = sample.weight;

    model_inputs::ModelInputView oldModelInputView = buildModelInputUpToObservation(previousObservationId, batchedTrainingInput);
    model_inputs::ModelInputView newModelInputView = buildModelInputUpToObservation(observationId, batchedTrainingInput);

    batchedTrainingInput.oldModelInputViews.push_back(std::move(oldModelInputView));
    batchedTrainingInput.actionsTaken.push_back(actionIndex0.value());
    batchedTrainingInput.isTerminals.push_back(isTerminal);
    batchedTrainingInput.rewards.push_back(reward);
    batchedTrainingInput.newModelInputViews.push_back(std::move(newModelInputView));
    batchedTrainingInput.importanceSamplingWeights.push_back(weight);
  }

  return batchedTrainingInput;
}

model_inputs::ModelInputView TrainingManager::buildModelInputUpToObservation(ObservationAndActionStorage::Id currentObservationId, model_inputs::BatchedTrainingInput &batchedTrainingInput) const {
  model_inputs::ModelInputView modelInput;
  modelInput.currentObservation = &batchedTrainingInput.getObservationAndAction(currentObservationId, observationAndActionStorage_).observation;
  modelInput.pastObservationStack.reserve(kPastObservationStackSize);
  {
    std::vector<std::pair<const Observation*, int>> pastObservationsAndActions;
    pastObservationsAndActions.reserve(kPastObservationStackSize);

    ObservationAndActionStorage::Id tmp = currentObservationId;
    while (pastObservationsAndActions.size() < kPastObservationStackSize && observationAndActionStorage_.hasPrevious(tmp)) {
      tmp = observationAndActionStorage_.getPreviousId(tmp);
      const ObservationAndActionStorage::ObservationAndActionType &obsAndAction = batchedTrainingInput.getObservationAndAction(tmp, observationAndActionStorage_);
      if (!obsAndAction.actionIndex) {
        throw std::runtime_error("Building stack of past observations, but have a missing action index");
      }
      pastObservationsAndActions.emplace_back(&obsAndAction.observation, obsAndAction.actionIndex.value());
    }

    // Add past observations in correct order (oldest first)
    for (auto it = pastObservationsAndActions.rbegin(); it != pastObservationsAndActions.rend(); ++it) {
      modelInput.pastObservationStack.push_back(it->first);
      modelInput.pastActionStack.push_back(it->second);
    }
  }
  return modelInput;
}

void TrainingManager::precompileModels() {
  // Create some spoof inputs so that we can invoke the model and trigger compilation.
  Observation observation;
  model_inputs::ModelInputView modelInputView;
  modelInputView.pastObservationStack.resize(kPastObservationStackSize, &observation);
  modelInputView.pastActionStack.resize(kPastObservationStackSize, 0);
  modelInputView.currentObservation = &observation;

  JaxInterface::Model dummyModel = jaxInterface_.getDummyModel();
  JaxInterface::Optimizer dummyOptimizer = jaxInterface_.getDummyOptimizer();
  JaxInterface::Model dummyTargetModel = jaxInterface_.getDummyModel();
  std::vector<model_inputs::ModelInputView> pastModelInputViews(kBatchSize, modelInputView);
  std::vector<int> actionsTaken(kBatchSize, 0);
  std::vector<bool> isTerminals(kBatchSize, false);
  std::vector<float> rewards(kBatchSize, 0.0f);
  std::vector<model_inputs::ModelInputView> currentModelInputViews(kBatchSize, modelInputView);
  std::vector<float> importanceSamplingWeights(kBatchSize, 0.0f);
  LOG(INFO) << "Precompiling selectAction";
  jaxInterface_.selectAction(modelInputView, /*canSendPacket=*/false);
  LOG(INFO) << "Precompiling train";
  JaxInterface::TrainAuxOutput trainOutput = jaxInterface_.train(dummyModel, dummyOptimizer, dummyTargetModel, pastModelInputViews, actionsTaken, isTerminals, rewards, currentModelInputViews, importanceSamplingWeights);
  LOG(INFO) << "Precompilation complete";
}

void TrainingManager::defineCharacterPairingsAndPositions() {
  constexpr int kAreaWidth = 20;
  constexpr int kAreaHeight = 50;
  // Define PVP positions
  int currentIndex = 0;
  for (int sum=0;; ++sum) {
    bool foundOne = false;
    for (int col=0; col<=sum && col<kAreaWidth; ++col) {
      const int row = sum - col;
      if (row >= kAreaHeight) {
        LOG(INFO) << "Skipping " << col+1 << ',' << row+1 << " as it is outside the defined area";
        continue;
      }
      LOG(INFO) << "Adding position at region (" << col+1 << "," << row+1 << ")";
      pvpPositions_.push_back({sro::Position(sro::position_math::worldRegionIdFromSectors(col+1, row+1), 960.0, 20.0, 960.0)});
      foundOne = true;
      ++currentIndex;
      if (currentIndex >= kPvpCount) {
        break;
      }
    }
    if (currentIndex >= kPvpCount) {
      break;
    }
    if (!foundOne) {
      // No more open positions.
      LOG(ERROR) << "Added only " << currentIndex << " positions. No more fit. We need " << kPvpCount << ". Stopping position generation.";
      break;
    }
  }

  for (int i=0; i<kPvpCount; ++i) {
    // Define character pairings
    LOG(INFO) << "Adding character pairing: " << absl::StrFormat("rl%d", i*2) << ',' << absl::StrFormat("RL_%d", i*2) << ',' << absl::StrFormat("rl%d", i*2+1) << ',' << absl::StrFormat("RL_%d", i*2+1);
    characterPairings_.push_back({
      CharacterLoginInfo{/*username=*/absl::StrFormat("rl%d", i*2),   /*password=*/"0", /*characterName=*/absl::StrFormat("RL_%d", i*2)},
      CharacterLoginInfo{/*username=*/absl::StrFormat("rl%d", i*2+1), /*password=*/"0", /*characterName=*/absl::StrFormat("RL_%d", i*2+1)},
      i
    });
  }

  // Verify we have at least as many positions as pairings
  if (pvpPositions_.size() < characterPairings_.size()) {
    throw std::runtime_error("Not enough PVP positions defined for all character pairings");
  }
}

void TrainingManager::checkAndPublishPvpDescriptors() {
  // Iterate through ready sessions and check if any pairs are ready
  for (auto& pairing : characterPairings_) {
    // Skip pairings with no sessions
    if (!pairing.session1Id || !pairing.session2Id) {
      continue;
    }

    // Check if both characters in this pairing are ready for assignment
    auto it1 = std::find(sessionsReadyForAssignment_.begin(), sessionsReadyForAssignment_.end(), pairing.session1Id.value());
    auto it2 = std::find(sessionsReadyForAssignment_.begin(), sessionsReadyForAssignment_.end(), pairing.session2Id.value());

    if (it1 != sessionsReadyForAssignment_.end() && it2 != sessionsReadyForAssignment_.end()) {
      // Both characters in this pairing are ready for assignment
      Session& char1 = getSession(pairing.session1Id.value());
      Session& char2 = getSession(pairing.session2Id.value());

      // Build the PVP descriptor
      common::PvpDescriptor pvpDescriptor = buildPvpDescriptor(char1, char2, pairing.positionIndex);

      // Remove these sessions from the ready list
      sessionsReadyForAssignment_.erase(it1);
      // Need to find it2 again as erasing it1 may have invalidated the iterator
      it2 = std::find(sessionsReadyForAssignment_.begin(), sessionsReadyForAssignment_.end(), pairing.session2Id.value());
      sessionsReadyForAssignment_.erase(it2);

      // Publish the PVP descriptor
      eventBroker_.publishEvent<event::BeginPvp>(std::move(pvpDescriptor));

      LOG(INFO) << "Published BeginPvp event for " << char1.getBot().selfState()->name << " & "
                << char2.getBot().selfState()->name << " at position index " << pairing.positionIndex
                << ". Now have " << sessionsReadyForAssignment_.size() << " sessions ready for assignment";
    }
  }
}

} // namespace rl