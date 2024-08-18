#include "bot.hpp"

#include "config/characterConfig.hpp"
#include "helpers.hpp"
#include "packet/building/clientAgentActionDeselectRequest.hpp"
#include "packet/building/clientAgentActionSelectRequest.hpp"
#include "packet/building/clientAgentActionTalkRequest.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"
#include "packet/building/clientAgentInventoryStorageOpenRequest.hpp"
// TODO: <remove>
// For quicker development, when we spawn in, set ourself as visible and put on a PVP cape
#include "packet/building/clientAgentFreePvpUpdateRequest.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"
#include "packet/building/clientAgentSkillMasteryLearnRequest.hpp"
// </remove>
#include "proto_convert/convert.hpp"
#include "state/machine/alchemy.hpp"
#include "state/machine/applyStatPoints.hpp"
#include "state/machine/autoPotion.hpp"
#include "state/machine/botting.hpp"
#include "state/machine/login.hpp"
#include "state/machine/maxMasteryAndSkills.hpp"
#include "type_id/categories.hpp"

#include "pathfinder.h"

#include <silkroad_lib/game_constants.h>
#include <silkroad_lib/position_math.h>

#include <absl/log/log.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>

#include <regex>

Bot::Bot(SessionId sessionId,
         const pk2::GameData &gameData,
         Proxy &proxy,
         broker::PacketBroker &packetBroker,
         broker::EventBroker &eventBroker,
         state::WorldState &worldState) :
      sessionId_(sessionId),
      gameData_(gameData),
      proxy_(proxy),
      packetBroker_(packetBroker),
      eventBroker_(eventBroker),
      worldState_(worldState) {
}

void Bot::initialize() {
  subscribeToEvents();
  packetProcessor_.initialize();
}

void Bot::setCharacterToLogin(std::string_view characterName) {
  loadConfig(characterName);
  if (config_) {
    config::LoginInfo loginInfo = config_->getLoginInfo();
    loginStateMachine_ = std::make_unique<state::machine::Login>(*this, loginInfo.username, loginInfo.password, characterName);
  }
}

void Bot::loadConfig(std::string_view characterName) {
  const auto appDataDirectory = helpers::getAppDataDirectory();
  try {
    config::CharacterConfig characterConfig;
    characterConfig.initialize(appDataDirectory, characterName);
    config_.emplace(std::move(characterConfig));
    VLOG(1) << "Character config for " << characterName << " successfully loaded";
  } catch (const std::exception& ex) {
    LOG(WARNING) << absl::StreamFormat("Error parsing character config for character \"%s\": \"%s\"", characterName, ex.what());
  }
}

const config::CharacterConfig* Bot::config() const {
  if (config_.has_value()) {
    return &config_.value();
  }
  return nullptr;
}

const pk2::GameData& Bot::gameData() const {
  return gameData_;
}

Proxy& Bot::proxy() const {
  return proxy_;
}

broker::PacketBroker& Bot::packetBroker() const {
  return packetBroker_;
}

broker::EventBroker& Bot::eventBroker() {
  return eventBroker_;
}

const state::WorldState& Bot::worldState() const {
  return worldState_;
}

state::EntityTracker& Bot::entityTracker() {
  return worldState_.entityTracker();
}

const state::EntityTracker& Bot::entityTracker() const {
  return worldState_.entityTracker();
}

std::shared_ptr<entity::Self> Bot::selfState() const {
  return selfEntity_;
}

void Bot::subscribeToEvents() {
  auto eventHandleFunction = std::bind(&Bot::handleEvent, this, std::placeholders::_1);
  // Bot actions from UI
  eventBroker_.subscribeToEvent(event::EventCode::kRequestStartTraining, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRequestStopTraining, eventHandleFunction);
  // Debug help
  eventBroker_.subscribeToEvent(event::EventCode::kInjectPacket, eventHandleFunction);
  // Login events
  eventBroker_.subscribeToEvent(event::EventCode::kShardListReceived, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kGatewayLoginResponseReceived, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kConnectedToAgentServer, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kIbuvChallengeReceived, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kServerAuthSuccess, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterListReceived, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterSelectionJoinSuccess, eventHandleFunction);
  // Movement events
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementBegan, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityEnteredGeometry, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityExitedGeometry, eventHandleFunction);
  // Character info events
  eventBroker_.subscribeToEvent(event::EventCode::kSelfSpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kItemUseFailed, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityHpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMaxHpMpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStatesChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterAvailableStatPointsUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStatsChanged, eventHandleFunction);

  // Misc
  eventBroker_.subscribeToEvent(event::EventCode::kEntityDeselected, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntitySelected, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kNpcTalkStart, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventoryUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStorageInitialized, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStorageUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRepairSuccessful, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntitySpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityDespawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityBodyStateChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityLifeStateChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kItemUseTimeout, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSkillCastTimeout, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityOwnershipRemoved, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockedBack, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockedDown, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockbackStunEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockdownStunEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMovementRequestTimedOut, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kItemCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventoryItemUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kHwanPointsUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kAlchemyCompleted, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kAlchemyTimedOut, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kGmCommandTimedOut, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kChatReceived, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kGameReset, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSetCurrentPositionAsTrainingCenter, eventHandleFunction);
  // eventBroker_.subscribeToEvent(event::EventCode::kNewConfigReceived, eventHandleFunction);
  // eventBroker_.subscribeToEvent(event::EventCode::kConfigUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kResurrectOption, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kLearnMasterySuccess, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kLearnSkillSuccess, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kLearnSkillError, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kTimeout, eventHandleFunction);

  // Skills
  eventBroker_.subscribeToEvent(event::EventCode::kSkillBegan, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSkillEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kOurSkillFailed, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kPlayerCharacterBuffAdded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kPlayerCharacterBuffRemoved, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kOurCommandError, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSkillCooldownEnded, eventHandleFunction);

  eventBroker_.subscribeToEvent(event::EventCode::kStateMachineActiveTooLong, eventHandleFunction);
}

void Bot::handleEvent(const event::Event *event) {
  std::unique_lock<std::mutex> worldStateLock(worldState_.mutex);
  try {
    // Do special handling for specific events before calling onUpdate.
    const auto eventCode = event->eventCode;
    switch (eventCode) {
      // Bot actions from UI
      case event::EventCode::kRequestStartTraining: {
        handleRequestStartTraining();
        break;
      }
      case event::EventCode::kRequestStopTraining: {
        handleRequestStopTraining();
        break;
      }

      // Debug help
      case event::EventCode::kInjectPacket: {
        const event::InjectPacket &castedEvent = dynamic_cast<const event::InjectPacket&>(*event);
        handleInjectPacket(castedEvent);
        break;
      }

      // Character info events
      case event::EventCode::kSelfSpawned: {
        handleSpawned(event);
        break;
      }

      // Misc
      case event::EventCode::kEntitySpawned: {
        const auto &castedEvent = dynamic_cast<const event::EntitySpawned&>(*event);
        entitySpawned(castedEvent);
        break;
      }
      case event::EventCode::kEntityBodyStateChanged: {
        const auto &castedEvent = dynamic_cast<const event::EntityBodyStateChanged&>(*event);
        handleBodyStateChanged(castedEvent);
        break;
      }
      case event::EventCode::kKnockbackStunEnded: {
        handleKnockbackStunEnded();
        break;
      }
      case event::EventCode::kKnockdownStunEnded: {
        handleKnockdownStunEnded();
        break;
      }
      case event::EventCode::kItemCooldownEnded: {
        const auto &castedEvent = dynamic_cast<const event::ItemCooldownEnded&>(*event);
        handleItemCooldownEnded(castedEvent);
        break;
      }

      // Skills
      case event::EventCode::kSkillEnded: {
        const auto &castedEvent = dynamic_cast<const event::SkillEnded&>(*event);
        handleSkillEnded(castedEvent);
        break;
      }
      case event::EventCode::kSkillCooldownEnded: {
        const auto &castedEvent = dynamic_cast<const event::SkillCooldownEnded&>(*event);
        handleSkillCooldownEnded(castedEvent);
        break;
      }
      case event::EventCode::kChatReceived: {
        const auto &castedEvent = dynamic_cast<const event::ChatReceived&>(*event);
        handleChatCommand(castedEvent);
        break;
      }
      case event::EventCode::kGameReset: {
        handleGameReset(event);
        break;
      }
      case event::EventCode::kSetCurrentPositionAsTrainingCenter: {
        setCurrentPositionAsTrainingCenter();
        break;
      }
      case event::EventCode::kLearnSkillSuccess: {
        const auto &castedEvent = dynamic_cast<const event::LearnSkillSuccess&>(*event);
        handleLearnedSkill(castedEvent);
        break;
      }
      default:
        break;
    }

    // Always call onUpdate for any event we receive.
    onUpdate(event);
  } catch (std::exception &ex) {
    LOG(INFO) << absl::StreamFormat("Error while handling event %s: \"%s\"", event::toString(event->eventCode), ex.what());
  }
}

// ============================================================================================================================
// ====================================================Main Logic Game Loop====================================================
// ============================================================================================================================

void Bot::onUpdate(const event::Event *event) {
  if (loginStateMachine_) {
    VLOG(1) << "Calling into login state machine " << sessionId() << " with event " << event::toString(event->eventCode);
    loginStateMachine_->onUpdate(event);
    if (loginStateMachine_->done()) {
      loginStateMachine_.reset();
    }
  }

  concurrentStateMachines_.onUpdate(event);

  // Highest priority is our vitals, we will try to heal even if we're not training
  if (autoPotionStateMachine_) {
    try {
      autoPotionStateMachine_->onUpdate(event);
      if (autoPotionStateMachine_->done()) {
        LOG(INFO) << "Autopotion is done. Resetting";
        autoPotionStateMachine_.reset();
      }
    } catch (std::exception &ex) {
      LOG(INFO) << "Error while running autopotion: " << ex.what();
    }
  }

  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (selfEntity && !selfEntity->trainingIsActive) {
    // Not training, nothing else to do
    return;
  }

  if (bottingStateMachine_ && !bottingStateMachine_->done()) {
    bottingStateMachine_->onUpdate(event);
  }
}

// ============================================================================================================================
// =====================================================Bot actions from UI====================================================
// ============================================================================================================================

void Bot::handleRequestStartTraining() {
  startTraining();
}

void Bot::handleRequestStopTraining() {
  stopTraining();
}

void Bot::startTraining() {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    LOG(WARNING) << "Tried to start training, but self is not spawned";
    return;
  }
  if (selfEntity->trainingIsActive) {
    LOG(INFO) << "Asked to start training, but we're already training";
    return;
  }

  if (bottingStateMachine_) {
    throw std::runtime_error("Asked to start training, but already have a botting state machine");
  }

  LOG(INFO) << "Starting training";
  selfEntity->trainingIsActive = true;
  eventBroker_.publishEvent(event::EventCode::kTrainingStarted);
  // TODO: Should we stop whatever we're doing?
  //  For example, if we're running, stop where we are.

  // Initialize state machine
  bottingStateMachine_ = std::make_unique<state::machine::Botting>(*this);
  // bottingStateMachine_ = std::make_unique<state::machine::Alchemy>(*this);
}

void Bot::stopTraining() {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    LOG(WARNING) << "Tried to stop training, but self is not spawned";
    return;
  }
  if (selfEntity->trainingIsActive) {
    // TODO: Need to cleanup current action to avoid leaving the client in a bad state
    //  Ex. Need to close a shop npc dialog
    LOG(INFO) << "Stopping training";
    selfEntity->trainingIsActive = false;
    eventBroker_.publishEvent(event::EventCode::kTrainingStopped);
    bottingStateMachine_.reset();
  } else {
    LOG(INFO) << "Asked to stop training, but we weren't training";
  }
}

// ============================================================================================================================
// =======================================================Chat commands========================================================
// ============================================================================================================================

void Bot::handleChatCommand(const event::ChatReceived &event) {
  std::regex commandRegex(R"delim(cmd (.*))delim", std::regex::ECMAScript);
  std::regex maxRegex(R"delim(max ([a-z\s]+))delim", std::regex::ECMAScript);
  if (event.chatType == packet::enums::ChatType::kAll ||
      event.chatType == packet::enums::ChatType::kAllGm) {
    // All chat
    std::string msg = event.message;
    std::transform(msg.begin(), msg.end(), msg.begin(), [](unsigned char c) { return std::tolower(c); });
    if (std::smatch commandMatchResults; std::regex_match(event.message, commandMatchResults, commandRegex)) {
      LOG(INFO) << "Received command: \"" << commandMatchResults.str(0) << '"';
      const std::string command = commandMatchResults.str(1);
      if (command == "start") {
        // Start training
        startTraining();
      } else if (command == "stop") {
        // Stop training
        stopTraining();
      } else if (std::smatch maxMatchResults; std::regex_match(command, maxMatchResults, maxRegex)) {
        const std::string maxMatch = maxMatchResults.str(1);
        std::vector<std::string_view> thingsToMax = absl::StrSplit(maxMatch, ' ');
        thingsToMax.erase(std::remove_if(thingsToMax.begin(), thingsToMax.end(), [](const auto &s) { return s.empty(); }), thingsToMax.end());
        LOG(INFO) << "    Want to max " << absl::StrJoin(thingsToMax, ",");
        for (std::string_view thingToMax : thingsToMax) {
          if (thingToMax == "str" || thingToMax == "int") {
            std::shared_ptr<entity::Self> selfEntity = selfState();
            if (!selfEntity) {
              // Self is not spawned.
              LOG(WARNING) << "Self is not spawned when trying to handle chat command";
              return;
            }
            if (selfEntity->getAvailableStatPoints() > 0) {
              const state::machine::StatPointType type = (thingToMax == "str" ? state::machine::StatPointType::kStr : state::machine::StatPointType::kInt);
              LOG(INFO) << "Have " << selfEntity->getAvailableStatPoints() << " stat points for " << static_cast<int>(type);
              concurrentStateMachines_.emplace<state::machine::ApplyStatPoints>(std::vector<state::machine::StatPointType>(selfEntity->getAvailableStatPoints(), type));
            } else {
              LOG(INFO) << "No available stat points for " << thingToMax;
            }
          } else {
            // Is this a mastery?
            const auto masteryId = gameData_.getMasteryId(std::string(thingToMax));
            LOG(INFO) << "Asking to max mastery ID " << masteryId << " = \"" << thingToMax << "\"";
            concurrentStateMachines_.emplace<state::machine::MaxMasteryAndSkills>(masteryId);
          }
        }
      }
    }
  }
}

// ============================================================================================================================
// =========================================================Debug help=========================================================
// ============================================================================================================================

void Bot::handleInjectPacket(const event::InjectPacket &castedEvent) {
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
  PacketContainer::Direction direction = (castedEvent.direction == event::InjectPacket::Direction::kClientToServer) ? PacketContainer::Direction::kClientToServer : PacketContainer::Direction::kServerToClient;
  StreamUtility stream;
  for (const auto i : castedEvent.data) {
    stream.Write<uint8_t>(i);
  }
  const auto packet = PacketContainer(static_cast<uint16_t>(castedEvent.opcode), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
  LOG(INFO) << "Injecting packet";
  packetBroker_.injectPacket(packet, direction);
}

// ============================================================================================================================
// ===============================================Character info packet handling===============================================
// ============================================================================================================================

void Bot::handleSpawned(const event::Event *event) {
  const event::SelfSpawned *selfSpawnedEvent = dynamic_cast<const event::SelfSpawned*>(event);
  if (selfSpawnedEvent == nullptr) {
    throw std::runtime_error("Failed to cast SelfSpawned event");
  }
  if (selfSpawnedEvent->sessionId != sessionId()) {
    // Not for us.
    return;
  }
  selfEntity_ = worldState_.getEntity<entity::Self>(selfSpawnedEvent->globalId);

  // Construct an autopotion state machine once the character is logged in.
  if (config_) {
    LOG(INFO) << "Constructing autopotion";
    autoPotionStateMachine_ = std::make_unique<state::machine::AutoPotion>(*this);
  } else {
    LOG(WARNING) << "Not constructing AutoPotion because we have no character config";
  }
}

// ============================================================================================================================
// ===========================================================Skills===========================================================
// ============================================================================================================================

void Bot::handleSkillEnded(const event::SkillEnded &event) {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    LOG(WARNING) << "Skill ended, but self is not spawned";
    return;
  }
  if (event.casterGlobalId != selfEntity->globalId) {
    // Skill ended, but it is not for us.
    return;
  }
  // LOG(INFO) << "Our skill \"" << gameData_.getSkillName(event.skillRefId) << "\" ended";
}

void Bot::handleSkillCooldownEnded(const event::SkillCooldownEnded &event) {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    LOG(WARNING) << "Skill cooldown ended, but self is not spawned";
    return;
  }
  // LOG(INFO) << "Skill " << event.skillRefId << "(" << gameData_.getSkillName(event.skillRefId) << ") cooldown ended";
  selfEntity->skillEngine.skillCooldownEnded(event.skillRefId);
}

// ============================================================================================================================
// ============================================================Misc============================================================
// ============================================================================================================================

void Bot::entitySpawned(const event::EntitySpawned &event) {
  const bool trackingEntity = worldState_.entityTracker().trackingEntity(event.globalId);
  if (!trackingEntity) {
    // TODO: Maybe we ought to move this check at the source of the event and get rid of this
    throw std::runtime_error(absl::StrFormat("Received entity spawned event, but we are not tracking entity %d", event.globalId));
  }
}

void Bot::handleBodyStateChanged(const event::EntityBodyStateChanged &event) {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    LOG(WARNING) << "Body state changed, but self is not spawned";
    return;
  }
  if (event.globalId == selfEntity->globalId) {
    // Our body state changed
    if (selfEntity->bodyState() == packet::enums::BodyState::kInvisibleGm) {
      // For quicker development, when we spawn in, set ourself as visible and put on a PVP cape
      // Set self as visible
      LOG(INFO) << "Setting self as visible";
      const auto setVisiblePacket = packet::building::ClientAgentOperatorRequest::toggleInvisible();
      packetBroker_.injectPacket(setVisiblePacket, PacketContainer::Direction::kClientToServer);

      // LOG(INFO) << "Setting free pvp mode";
      // const auto setPvpModePacket = packet::building::ClientAgentFreePvpUpdateRequest::setMode(packet::enums::FreePvpMode::kYellow);
      // packetBroker_.injectPacket(setPvpModePacket, PacketContainer::Direction::kClientToServer);
    }
  }
}

void Bot::handleKnockbackStunEnded() {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    LOG(WARNING) << "Knockback stun ended, but self is not spawned";
    return;
  }
  selfEntity->stunnedFromKnockback = false;
}

void Bot::handleKnockdownStunEnded() {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    LOG(WARNING) << "Knockdown stun ended, but self is not spawned";
    return;
  }
  selfEntity->stunnedFromKnockdown = false;
}

void Bot::handleItemCooldownEnded(const event::ItemCooldownEnded &event) {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    LOG(WARNING) << "Item cooldown ended, but self is not spawned";
    return;
  }
  if (event.globalId == selfEntity->globalId) {
    selfEntity->itemCooldownEnded(event.typeId);
  }
}

void Bot::handleGameReset(const event::Event *event) {
  LOG(INFO) << "Bot stops tracking self";
  selfEntity_.reset();
}

void Bot::setCurrentPositionAsTrainingCenter() {
  // proto::config::CharacterConfig *characterConfig = config_.getCharacterConfig(selfState().name);
  // proto::config::TrainingConfig *trainingConfig = characterConfig->mutable_training_config();
  // const auto currentPosition = selfState().position();
  // proto_convert::positionToProto(currentPosition, *trainingConfig->mutable_center());
  // config_.save();
  // eventBroker_.publishEvent<event::ConfigUpdated>(config_.configProto());
}

void Bot::handleLearnedSkill(const event::LearnSkillSuccess &event) {
  // if (!event.oldSkillRefId.has_value()) {
  //   // This is the first skill in this group, it can not exist in the config.
  //   // TODO: Skills in different groups should overwrite others. For example, the first book of fire imbue and second book are different groups, but once we get the second book, it should(?) overwrite.
  //   return;
  // }
  // // If this skill is in our config, update it with the new one.
  // auto *characterConfig = config_.getCharacterConfig(selfState().name);
  // if (characterConfig == nullptr) {
  //   return;
  // }
  // auto *trainingConfig = characterConfig->mutable_training_config();
  // bool updatedConfig=false;
  // {
  //   auto *attackSkills = trainingConfig->mutable_training_attack_skill_ids();
  //   auto it = std::find(attackSkills->begin(), attackSkills->end(), *event.oldSkillRefId);
  //   if (it != attackSkills->end()) {
  //     *it = event.newSkillRefId;
  //     updatedConfig = true;
  //   }
  // }
  // {
  //   auto *trainingBuffSkills = trainingConfig->mutable_training_buff_skill_ids();
  //   auto it = std::find(trainingBuffSkills->begin(), trainingBuffSkills->end(), *event.oldSkillRefId);
  //   if (it != trainingBuffSkills->end()) {
  //     *it = event.newSkillRefId;
  //     updatedConfig = true;
  //   }
  // }
  // {
  //   auto *nonTrainingBuffSkills = trainingConfig->mutable_nontraining_buff_skill_ids();
  //   auto it = std::find(nonTrainingBuffSkills->begin(), nonTrainingBuffSkills->end(), *event.oldSkillRefId);
  //   if (it != nonTrainingBuffSkills->end()) {
  //     *it = event.newSkillRefId;
  //     updatedConfig = true;
  //   }
  // }
  // if (updatedConfig) {
  //   config_.save();
  //   eventBroker_.publishEvent<event::ConfigUpdated>(config_.configProto());
  // }
}

bool Bot::needToGoToTown() const {
  // TODO: Get this from configuration. Maybe have it be implemented as a list of ReturnToTownCondition functors
  // TODO: Since we're actually botting such a low level character, we wont send him to town for not having enough potions.
  // const auto mpPotionSlots = selfState().inventory.findItemsOfCategory({type_id::categories::kMpPotion});
  // if (mpPotionSlots.empty()) {
  //   LOG(INFO) << "Checking if we need to go to town. Have no MP potions";
  //   return true;
  // }
  // const auto hpPotionSlots = selfState().inventory.findItemsOfCategory({type_id::categories::kHpPotion});
  // if (hpPotionSlots.empty()) {
  //   LOG(INFO) << "Checking if we need to go to town. Have no HP potions";
  //   return true;
  // }
  // int hpPotionCount=0;
  // for (const auto slot : hpPotionSlots) {
  //   const auto *item = selfState().inventory.getItem(slot);
  //   if (item == nullptr) {
  //     throw std::runtime_error("Expecting hp potion, but item is null");
  //   }
  //   const auto *itemAsExp = dynamic_cast<const storage::ItemExpendable*>(item);
  //   if (itemAsExp == nullptr) {
  //     throw std::runtime_error("Expecting hp potion, but expendable is null");
  //   }
  //   hpPotionCount += itemAsExp->quantity;
  // }
  // constexpr const int kMinHpCount{10};
  // if (hpPotionCount < kMinHpCount) {
  //   LOG(INFO) << "Checking if we need to go to town. Have fewer than " << kMinHpCount << " HP potions";
  //   return true;
  // }
  return false;
}

bool Bot::similarSkillIsAlreadyActive(sro::scalar_types::ReferenceObjectId skillRefId) const {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    throw std::runtime_error("Trying to check if similar skill is active, but self is not spawned");
  }
  const auto &givenSkillData = gameData_.skillData().getSkillById(skillRefId);
  const auto activeBuffs = selfEntity->activeBuffs();
  for (const auto activeBuffId : activeBuffs) {
    const auto &activeBuffData = gameData_.skillData().getSkillById(activeBuffId);
    if (givenSkillData.actionOverlap == activeBuffData.actionOverlap) {
      // These two cannot be active at the same time
      return true;
    }
    if (givenSkillData.hasParam(pk2::ref::skill_param::kHaste) &&
        activeBuffData.hasParam(pk2::ref::skill_param::kHaste)) {
      return true;
    }
  }
  return false;
}

bool Bot::canCastSkill(sro::scalar_types::ReferenceObjectId skillRefId) const {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    throw std::runtime_error("Trying to check if we can cast a skill, but self is not spawned");
  }
  const auto &skillData = gameData().skillData().getSkillById(skillRefId);
  // TODO: Keep track if we're wearing a full protector or garment set and reduce the MP requirement by 10%/20% respectively.
  if (skillData.consumeMP > selfEntity->currentMp() ||
      (selfEntity->maxMp() && skillData.consumeMPRatio > (static_cast<double>(selfEntity->currentMp()) / *selfEntity->maxMp()) * 100)) {
    // Not enough MP to cast.
    LOG(INFO) << "Not enough MP to cast skill " << gameData().getSkillName(skillRefId);
    return false;
  }
  const auto currentHp = selfEntity->currentHp();
  if (skillData.consumeHP > currentHp ||
      (selfEntity->maxHp() && skillData.consumeHPRatio > (static_cast<double>(currentHp) / *selfEntity->maxHp()) * 100)) {
    // Not enough HP to cast.
    LOG(INFO) << "Not enough HP to cast skill " << gameData().getSkillName(skillRefId);
    return false;
  }
  if (selfEntity->skillEngine.skillIsOnCooldown(skillRefId)) {
    return false;
  }
  if (selfEntity->stunnedFromKnockback || selfEntity->stunnedFromKnockdown) {
    // Stunned from KB or KD, cannot use this skill
    // TODO: Maybe there are some skills which can be used while knocked down
    return false;
  }
  return true;
}

std::vector<packet::building::NetworkReadyPosition> Bot::calculatePathToDestination(const sro::Position &destinationPosition) const {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    throw std::runtime_error("Trying to calculate path to destination, but self is not spawned");
  }
  // Since we can only move to positions on whole integers, find the closest possible point to the destination position while also accounting for the transformation that happens to the packet while being converted to be sent over the network
  const auto networkReadyPos = packet::building::NetworkReadyPosition::roundToNearest(destinationPosition);
  const sro::Position closestDestinationPosition = networkReadyPos.asSroPosition();
  if (sro::position_math::calculateDistance2d(selfEntity->position(), closestDestinationPosition) <= sqrt(0.5)) {
    // Already at destination
    return {};
  }
  auto pathfindingResultPathToVectorOfPositions = [&](const auto &pathfindingShortestPath) {
    const auto &navmeshTriangulation = gameData().navmeshTriangulation();

    // Get a list of all straight segments
    std::vector<pathfinder::StraightPathSegment*> straightSegments;
    for (const auto &segment : pathfindingShortestPath) {
      pathfinder::StraightPathSegment *straightSegment = dynamic_cast<pathfinder::StraightPathSegment*>(segment.get());
      if (straightSegment != nullptr) {
        straightSegments.push_back(straightSegment);
      }
    }

    // Turn straight segments into a list of waypoints
    std::vector<sro::Position> waypoints;
    // Note: We are ignoring the start of the first segment, since we assume we're already there
    for (int i=0; i<straightSegments.size()-1; ++i) {
      // Find the average between the end of this straight segment and the beginning of the next
      //  Between these two is an arc, which we're ignoring
      // TODO: There is a chance that this yields a bad path
      const auto &point1 = straightSegments[i]->endPoint;
      const auto &point2 = straightSegments[i+1]->startPoint;
      const auto midpoint = pathfinder::math::extendLineSegmentToLength(point1, point2, pathfinder::math::distance(point1, point2)/2.0);
      const auto regionAndPointPair = navmeshTriangulation.transformAbsolutePointIntoRegion({static_cast<float>(midpoint.x()), 0.0f, static_cast<float>(midpoint.y())});
      // TODO: Rounding the position could result in an invalid path
      waypoints.emplace_back(regionAndPointPair.first, std::round(regionAndPointPair.second.x), std::round(regionAndPointPair.second.y), std::round(regionAndPointPair.second.z));
    }

    // Additionally, add the endpoint of the final segment
    const auto finalRegionAndPointPair = navmeshTriangulation.transformAbsolutePointIntoRegion({static_cast<float>(straightSegments.back()->endPoint.x()), 0.0f, static_cast<float>(straightSegments.back()->endPoint.y())});
    waypoints.emplace_back(finalRegionAndPointPair.first, std::round(finalRegionAndPointPair.second.x), std::round(finalRegionAndPointPair.second.y), std::round(finalRegionAndPointPair.second.z));

    // Remove duplicates
    auto newEndIt = std::unique(waypoints.begin(), waypoints.end());
    if (newEndIt != waypoints.end()) {
      waypoints.erase(newEndIt, waypoints.end());
    }
    return waypoints;
  };

  auto breakUpLongMovements = [](std::vector<sro::Position> &waypoints) {
    auto tooFar = [](const auto &srcWaypoint, const auto &destWaypoint) {
      // The difference between a pair of xOffsets must be <= 1920.
      // The difference between a pair of zOffsets must be <= 1920.
      const auto [dx, dz] = sro::position_math::calculateOffset2d(srcWaypoint, destWaypoint);
      return (std::abs(dx) > sro::game_constants::kRegionWidth ||
              std::abs(dz) > sro::game_constants::kRegionHeight);
    };
    auto splitWaypoints = [](const auto &srcWaypoint, const auto &destWaypoint) {
      const auto [dx, dz] = sro::position_math::calculateOffset2d(srcWaypoint, destWaypoint);
      if (std::abs(dx) > std::abs(dz)) {
        const double ratio = static_cast<double>(sro::game_constants::kRegionWidth) / std::abs(dx);
        const auto newDxOffset = (dx > 0 ? 1 : -1) * sro::game_constants::kRegionWidth;
        const double newDzOffset = dz * ratio;
        return sro::position_math::createNewPositionWith2dOffset(destWaypoint, -newDxOffset, -newDzOffset);
      } else {
        const double ratio = static_cast<double>(sro::game_constants::kRegionWidth) / std::abs(dz);
        const auto newDxOffset = dx * ratio;
        const double newDzOffset = (dz > 0 ? 1 : -1) * sro::game_constants::kRegionWidth;
        return sro::position_math::createNewPositionWith2dOffset(destWaypoint, -newDxOffset, -newDzOffset);
      }
    };
    for (int i=waypoints.size()-1; i>0;) {
      if (tooFar(waypoints.at(i-1), waypoints.at(i))) {
        // Pick a point that is the maximum distance possible away from waypoints[i] and insert it before waypoints[i]
        const auto newWaypoint = splitWaypoints(waypoints.at(i-1), waypoints.at(i));
        waypoints.insert(waypoints.begin()+i, newWaypoint);
      } else {
        --i;
      }
    }
  };

  auto convertWaypointsToNetworkReadyPoints = [](const std::vector<sro::Position> &waypoints) {
    std::vector<packet::building::NetworkReadyPosition> result;
    result.reserve(waypoints.size());
    for (const auto &pos : waypoints) {
      result.emplace_back(packet::building::NetworkReadyPosition::roundToNearest(pos));
    }
    return result;
  };

  constexpr const double kAgentRadius{3.14};
  pathfinder::Pathfinder<sro::navmesh::triangulation::NavmeshTriangulation> pathfinder(gameData().navmeshTriangulation(), kAgentRadius);
  pathfinder.setTimeout(std::chrono::milliseconds(150));
  std::string debugPath;
  try {
    const sro::Position currentPosition = selfEntity->position();
    const sro::math::Vector3 currentPositionPoint(currentPosition.xOffset(), currentPosition.yOffset(), currentPosition.zOffset());
    const auto navmeshCurrentPosition = gameData().navmeshTriangulation().transformRegionPointIntoAbsolute(currentPositionPoint, currentPosition.regionId());

    const sro::math::Vector3 destinationPositionPoint(closestDestinationPosition.xOffset(), closestDestinationPosition.yOffset(), closestDestinationPosition.zOffset());
    const auto navmeshDestinationPosition = gameData().navmeshTriangulation().transformRegionPointIntoAbsolute(destinationPositionPoint, destinationPosition.regionId());

    // TODO: If the src or dest positions are overlapping with a constraint, we need to add an extra point.
    debugPath = absl::StrFormat("Calculating path from (%d;%.10f,%.10f,%.10f) to (%d;%.10f,%.10f,%.10f)", currentPosition.regionId(), currentPosition.xOffset(), currentPosition.yOffset(), currentPosition.zOffset(), closestDestinationPosition.regionId(), closestDestinationPosition.xOffset(), closestDestinationPosition.yOffset(), closestDestinationPosition.zOffset());
    const auto pathfindingResult = pathfinder.findShortestPath(navmeshCurrentPosition, navmeshDestinationPosition);
    const auto &path = pathfindingResult.shortestPath;
    if (path.empty()) {
      throw std::runtime_error("Found empty path");
    }
    auto waypoints = pathfindingResultPathToVectorOfPositions(path);
    // Add our own position to the beginning of this list so that we can break up the distance if it's too far.
    waypoints.insert(waypoints.begin(), currentPosition);
    breakUpLongMovements(waypoints);
    return convertWaypointsToNetworkReadyPoints(waypoints);
  } catch (std::exception &ex) {
    LOG(INFO) << "Failed trying to build path: " << debugPath;
    throw std::runtime_error("Cannot find path with pathfinder: \""+std::string(ex.what())+"\"");
  }
}

sro::scalar_types::EntityGlobalId Bot::getClosestNpcGlobalId() const {
  std::shared_ptr<entity::Self> selfEntity = selfState();
  if (!selfEntity) {
    throw std::runtime_error("Trying to get closest NPC global ID, but self is not spawned");
  }
  std::optional<uint32_t> closestNpcGId;
  float closestNpcDistance = std::numeric_limits<float>::max();
  const auto &ourCurrentPosition = selfEntity->position();
  const auto &entityMap = entityTracker().getEntityMap();
  for (const auto &entityIdObjectPair : entityMap) {
    const auto &entityPtr = entityIdObjectPair.second;
    if (!entityPtr) {
      throw std::runtime_error("Entity map contains a null item");
    }

    if (entityPtr->entityType() != entity::EntityType::kNonplayerCharacter) {
      // Not an npc, skip
      continue;
    }

    const auto distanceToNpc = sro::position_math::calculateDistance2d(ourCurrentPosition, entityPtr->position());
    if (distanceToNpc < closestNpcDistance) {
      closestNpcGId = entityIdObjectPair.first;
      closestNpcDistance = distanceToNpc;
    }
  }
  if (!closestNpcGId) {
    throw std::runtime_error("There is no NPC within range, weird");
  }
  return *closestNpcGId;
}