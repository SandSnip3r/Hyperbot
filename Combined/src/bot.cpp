#include "bot.hpp"
#include "logging.hpp"

#include "packet/building/clientAgentActionDeselectRequest.hpp"
#include "packet/building/clientAgentActionSelectRequest.hpp"
#include "packet/building/clientAgentActionTalkRequest.hpp"
#include "packet/building/clientAgentAuthRequest.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"
#include "packet/building/clientAgentCharacterSelectionJoinRequest.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"
#include "packet/building/clientAgentInventoryStorageOpenRequest.hpp"
#include "packet/building/clientGatewayLoginIbuvAnswer.hpp"
#include "packet/building/clientGatewayLoginRequest.hpp"

#include <silkroad_lib/position_math.h>

Bot::Bot(const config::CharacterLoginData &loginData,
         const pk2::GameData &gameData,
         Proxy &proxy,
         broker::PacketBroker &packetBroker,
         broker::EventBroker &eventBroker) :
      loginData_(loginData),
      gameData_(gameData),
      proxy_(proxy),
      packetBroker_(packetBroker),
      eventBroker_(eventBroker) {
  userInterface_.run();
  subscribeToEvents();
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

state::EntityTracker& Bot::entityTracker() {
  return worldState_.entityTracker();
}

state::Self& Bot::selfState() {
  return worldState_.selfState();
}

void Bot::subscribeToEvents() {
  auto eventHandleFunction = std::bind(&Bot::handleEvent, this, std::placeholders::_1);
  // Bot actions from UI
  eventBroker_.subscribeToEvent(event::EventCode::kStartTraining, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStopTraining, eventHandleFunction);
  // Debug help
  eventBroker_.subscribeToEvent(event::EventCode::kInjectPacket, eventHandleFunction);
  // Login events
  eventBroker_.subscribeToEvent(event::EventCode::kStateShardIdUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateConnectedToAgentServerUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateReceivedCaptchaPromptUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kLoggedIn, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateCharacterListUpdated, eventHandleFunction);
  // Movement events
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementBegan, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementTimerEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityPositionUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityNotMovingAngleChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEnteredNewRegion, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityEnteredGeometry, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityExitedGeometry, eventHandleFunction);
  // Character info events
  eventBroker_.subscribeToEvent(event::EventCode::kSpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCosSpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kItemUseFailed, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityHpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMaxHpMpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStatesChanged, eventHandleFunction);

  // Misc
  eventBroker_.subscribeToEvent(event::EventCode::kEntityDeselected, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntitySelected, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kNpcTalkStart, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventoryUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kAvatarInventoryUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCosInventoryUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStorageInitialized, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kGuildStorageInitialized, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStorageUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kGuildStorageUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRepairSuccessful, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventoryGoldUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStorageGoldUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kGuildStorageGoldUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterSkillPointsUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterExperienceUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntitySpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityDespawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityLifeStateChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kTrainingAreaSet, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kTrainingAreaReset, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kItemUseTimeout, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityOwnershipRemoved, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateMachineCreated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateMachineDestroyed, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockedBack, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockedDown, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockbackStunEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockdownStunEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMovementRequestTimedOut, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kItemCooldownEnded, eventHandleFunction);

  // Skills
  eventBroker_.subscribeToEvent(event::EventCode::kSkillBegan, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSkillEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kOurSkillFailed, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kOurBuffRemoved, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kOurCommandError, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSkillCooldownEnded, eventHandleFunction);
}

void Bot::handleEvent(const event::Event *event) {
  std::unique_lock<std::mutex> selfStateLock(worldState_.selfState().selfMutex);

  try {
    const auto eventCode = event->eventCode;
    switch (eventCode) {
      // Bot actions from UI
      case event::EventCode::kStartTraining:
        handleStartTraining();
        break;
      case event::EventCode::kStopTraining:
        handleStopTraining();
        break;

      // Debug help
      case event::EventCode::kInjectPacket:
        {
          const event::InjectPacket &castedEvent = dynamic_cast<const event::InjectPacket&>(*event);
          handleInjectPacket(castedEvent);
        }
        break;

      // Login events
      case event::EventCode::kStateShardIdUpdated:
        handleStateShardIdUpdated();
        break;
      case event::EventCode::kStateConnectedToAgentServerUpdated:
        handleStateConnectedToAgentServerUpdated();
        break;
      case event::EventCode::kStateReceivedCaptchaPromptUpdated:
        handleStateReceivedCaptchaPromptUpdated();
        break;
      case event::EventCode::kLoggedIn:
        handleLoggedIn();
        break;
      case event::EventCode::kStateCharacterListUpdated:
        handleStateCharacterListUpdated();
        break;

      // Movement events
      case event::EventCode::kEntityMovementBegan:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityMovementBegan&>(*event);
          handleEntityMovementBegan(castedEvent);
          break;
        }
      case event::EventCode::kEntityMovementEnded:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityMovementEnded&>(*event);
          handleEntityMovementEnded(castedEvent);
          break;
        }
      case event::EventCode::kEntityMovementTimerEnded:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityMovementTimerEnded&>(*event);
          handleEntityMovementTimerEnded(castedEvent.globalId);
          break;
        }
      case event::EventCode::kEntityPositionUpdated:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityPositionUpdated&>(*event);
          handleEntityPositionUpdated(castedEvent.globalId);
          break;
        }
      case event::EventCode::kEntityNotMovingAngleChanged:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityNotMovingAngleChanged&>(*event);
          handleEntityNotMovingAngleChanged(castedEvent.globalId);
          break;
        }
      case event::EventCode::kEnteredNewRegion:
        {
          const auto pos = worldState_.selfState().position();
          const auto &regionName = gameData_.textZoneNameData().getRegionName(pos.regionId());
          userInterface_.broadcastRegionNameUpdate(regionName);
        }
        break;
      case event::EventCode::kEntityEnteredGeometry:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityEnteredGeometry&>(*event);
          handleEntityEnteredGeometry(castedEvent);
        }
        break;
      case event::EventCode::kEntityExitedGeometry:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityExitedGeometry&>(*event);
          handleEntityExitedGeometry(castedEvent);
        }
        break;

      // Character info events
      case event::EventCode::kSpawned:
        handleSpawned();
        break;
      case event::EventCode::kCosSpawned:
        {
          const event::CosSpawned &castedEvent = dynamic_cast<const event::CosSpawned&>(*event);
          handleCosSpawned(castedEvent);
        }
        break;
      case event::EventCode::kItemUseFailed:
        {
          onUpdate(event);
        }
        break;
      case event::EventCode::kEntityHpChanged: {
        const event::EntityHpChanged &castedEvent = dynamic_cast<const event::EntityHpChanged&>(*event);
        if (castedEvent.globalId == worldState_.selfState().globalId) {
          userInterface_.broadcastCharacterHpUpdate(worldState_.selfState().currentHp());
          handleVitalsChanged();
        }
        break;
      }
      case event::EventCode::kMpChanged:
        userInterface_.broadcastCharacterMpUpdate(worldState_.selfState().currentMp());
        handleVitalsChanged();
        break;
      case event::EventCode::kMaxHpMpChanged:
        if (worldState_.selfState().maxHp() && worldState_.selfState().maxMp()) {
          userInterface_.broadcastCharacterMaxHpMpUpdate(*worldState_.selfState().maxHp(), *worldState_.selfState().maxMp());
        }
        handleVitalsChanged();
        break;
      case event::EventCode::kStatesChanged:
        handleStatesChanged();
        break;

      // Misc
      case event::EventCode::kStorageInitialized:
        storageInitialized();
      case event::EventCode::kGuildStorageInitialized:
        guildStorageInitialized();
      case event::EventCode::kEntityDeselected:
      case event::EventCode::kEntitySelected:
      case event::EventCode::kNpcTalkStart:
      case event::EventCode::kRepairSuccessful:
        onUpdate();
        break;
      case event::EventCode::kInventoryUpdated:
        {
          const event::InventoryUpdated &castedEvent = dynamic_cast<const event::InventoryUpdated&>(*event);
          inventoryUpdated(castedEvent);
          break;
        }
      case event::EventCode::kAvatarInventoryUpdated:
        {
          const event::AvatarInventoryUpdated &castedEvent = dynamic_cast<const event::AvatarInventoryUpdated&>(*event);
          avatarInventoryUpdated(castedEvent);
          break;
        }
      case event::EventCode::kCosInventoryUpdated:
        {
          const event::CosInventoryUpdated &castedEvent = dynamic_cast<const event::CosInventoryUpdated&>(*event);
          cosInventoryUpdated(castedEvent);
          break;
        }
      case event::EventCode::kStorageUpdated:
        {
          const event::StorageUpdated &castedEvent = dynamic_cast<const event::StorageUpdated&>(*event);
          storageUpdated(castedEvent);
          break;
        }
      case event::EventCode::kGuildStorageUpdated:
        {
          const event::GuildStorageUpdated &castedEvent = dynamic_cast<const event::GuildStorageUpdated&>(*event);
          guildStorageUpdated(castedEvent);
          break;
        }
      case event::EventCode::kInventoryGoldUpdated:
        userInterface_.broadcastGoldAmountUpdate(worldState_.selfState().getGold(), broadcast::ItemLocation::kCharacterInventory);
        break;
      case event::EventCode::kStorageGoldUpdated:
        userInterface_.broadcastGoldAmountUpdate(worldState_.selfState().getStorageGold(), broadcast::ItemLocation::kStorage);
        break;
      case event::EventCode::kGuildStorageGoldUpdated:
        userInterface_.broadcastGoldAmountUpdate(worldState_.selfState().getGuildStorageGold(), broadcast::ItemLocation::kGuildStorage);
        break;
      case event::EventCode::kCharacterSkillPointsUpdated:
        userInterface_.broadcastCharacterSpUpdate(worldState_.selfState().getSkillPoints());
        break;
      case event::EventCode::kCharacterExperienceUpdated:
        userInterface_.broadcastCharacterExperienceUpdate(worldState_.selfState().getCurrentExperience(), worldState_.selfState().getCurrentSpExperience());
        break;
      case event::EventCode::kEntitySpawned:
        {
          const auto &castedEvent = dynamic_cast<const event::EntitySpawned&>(*event);
          entitySpawned(castedEvent);
          break;
        }
      case event::EventCode::kEntityDespawned:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityDespawned&>(*event);
          entityDespawned(castedEvent);
          break;
        }
      case event::EventCode::kEntityLifeStateChanged:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityLifeStateChanged&>(*event);
          entityLifeStateChanged(castedEvent);
          break;
        }
      case event::EventCode::kTrainingAreaSet:
        {
          if (!worldState_.selfState().trainingAreaGeometry) {
            throw std::runtime_error("Training area set, but no training geometry");
          }
          userInterface_.broadcastTrainingAreaSet(worldState_.selfState().trainingAreaGeometry.get());
          break;
        }
      case event::EventCode::kTrainingAreaReset:
        {
          LOG() << "Training area reset" << std::endl;
          userInterface_.broadcastTrainingAreaReset();
          break;
        }
      case event::EventCode::kItemUseTimeout:
        {
          const auto &castedEvent = dynamic_cast<const event::ItemUseTimeout&>(*event);
          itemUseTimedOut(castedEvent);
          break;
        }
      case event::EventCode::kEntityOwnershipRemoved:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityOwnershipRemoved&>(*event);
          onUpdate();
          break;
        }
      case event::EventCode::kStateMachineCreated:
        {
          const auto &castedEvent = dynamic_cast<const event::StateMachineCreated&>(*event);
          userInterface_.broadcastStateMachineCreated(castedEvent.stateMachineName);
          break;
        }
      case event::EventCode::kStateMachineDestroyed:
        {
          userInterface_.broadcastStateMachineDestroyed();
          break;
        }
      case event::EventCode::kKnockedBack:
        {
          onUpdate(event);
          break;
        }
      case event::EventCode::kKnockedDown:
        {
          onUpdate(event);
          break;
        }
      case event::EventCode::kKnockbackStunEnded:
        {
          handleKnockbackStunEnded();
          break;
        }
      case event::EventCode::kKnockdownStunEnded:
        {
          handleKnockdownStunEnded();
          break;
        }
      case event::EventCode::kMovementRequestTimedOut:
        {
          onUpdate(event);
          break;
        }
      case event::EventCode::kItemCooldownEnded:
        {
          const auto &castedEvent = dynamic_cast<const event::ItemCooldownEnded&>(*event);
          handleItemCooldownEnded(castedEvent);
          break;
        }

      // Skills
      case event::EventCode::kSkillBegan:
        {
          const auto &castedEvent = dynamic_cast<const event::SkillBegan&>(*event);
          handleSkillBegan(castedEvent);
          break;
        }
      case event::EventCode::kSkillEnded:
        {
          const auto &castedEvent = dynamic_cast<const event::SkillEnded&>(*event);
          handleSkillEnded(castedEvent);
          break;
        }
      case event::EventCode::kOurSkillFailed:
        {
          onUpdate(event);
          break;
        }
      case event::EventCode::kOurBuffRemoved:
        {
          onUpdate(event);
          break;
        }
      case event::EventCode::kOurCommandError:
        {
          onUpdate(event);
          break;
        }
      case event::EventCode::kSkillCooldownEnded:
        {
          const auto &castedEvent = dynamic_cast<const event::SkillCooldownEnded&>(*event);
          handleSkillCooldownEnded(castedEvent);
          break;
        }
      default:
        LOG() << "Unhandled event subscribed to. Code:" << static_cast<int>(eventCode) << '\n';
        break;
    }
  } catch (std::exception &ex) {
    LOG() << "Error while handling event!\n  " << ex.what() << std::endl;
  }
}

// ============================================================================================================================
// ====================================================Main Logic Game Loop====================================================
// ============================================================================================================================

void Bot::onUpdate(const event::Event *event) {
  // Highest priority is our vitals, we will try to heal even if we're not training
  autoPotionStateMachine_.onUpdate(event);

  if (!worldState_.selfState().trainingIsActive) {
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

void Bot::handleStartTraining() {
  LOG() << "Received message from UI! Start Training" << std::endl;
  startTraining();
}

void Bot::handleStopTraining() {
  LOG() << "Received message from UI! Stop Training" << std::endl;
  stopTraining();
}

void Bot::startTraining() {
  if (worldState_.selfState().trainingIsActive) {
    LOG() << "Asked to start training, but we're already training" << std::endl;
    return;
  }

  if (bottingStateMachine_) {
    throw std::runtime_error("Asked to start training, but already have a botting state machine");
  }

  worldState_.selfState().trainingIsActive = true;
  // TODO: Should we stop whatever we're doing?
  //  For example, if we're running, stop where we are

  // Initialize state machine
  bottingStateMachine_ = std::make_unique<state::machine::Botting>(*this);

  // Trigger onUpdate
  onUpdate();
}

void Bot::stopTraining() {
  if (worldState_.selfState().trainingIsActive) {
    // TODO: Need to cleanup current action to avoid leaving the client in a bad state
    //  Ex. Need to close a shop npc dialog
    LOG() << "Stopping training" << std::endl;
    worldState_.selfState().trainingIsActive = false;
    bottingStateMachine_.reset();
  } else {
    LOG() << "Asked to stop training, but we werent training" << std::endl;
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
  LOG() << "Injecting packet" << std::endl;
  packetBroker_.injectPacket(packet, direction);
}

// ============================================================================================================================
// ================================================Login process event handling================================================
// ============================================================================================================================

void Bot::handleStateShardIdUpdated() const {
  // We received the server list from the server, try to log in
  const auto loginAuthPacket = packet::building::ClientGatewayLoginRequest::packet(gameData_.divisionInfo().locale, loginData_.id, loginData_.password, worldState_.selfState().shardId);
  packetBroker_.injectPacket(loginAuthPacket, PacketContainer::Direction::kClientToServer);
}

void Bot::handleStateConnectedToAgentServerUpdated() {
  // Client will try to send auth request, block it
  if (proxy_.blockingOpcode(packet::Opcode::kClientAgentAuthRequest)) {
    throw std::runtime_error("Just connected to AgentServer, but something else blocked client auth packets");
  }
  proxy_.blockOpcode(packet::Opcode::kClientAgentAuthRequest);
  // Send our auth packet
  const auto clientAuthPacket = packet::building::ClientAgentAuthRequest::packet(worldState_.selfState().token, loginData_.id, loginData_.password, gameData_.divisionInfo().locale, worldState_.selfState().kMacAddress);
  packetBroker_.injectPacket(clientAuthPacket, PacketContainer::Direction::kClientToServer);
  // Set our state to logging in so that we'll know to block packets from the client if it tries to also login
  worldState_.selfState().loggingIn = true;
}

void Bot::handleLoggedIn() {
  worldState_.selfState().loggingIn = false;
  // Unblock packet
  if (!proxy_.blockingOpcode(packet::Opcode::kClientAgentAuthRequest)) {
    throw std::runtime_error("Just logged in, but we werent blocking the client's auth request");
  }
  proxy_.unblockOpcode(packet::Opcode::kClientAgentAuthRequest);
}

void Bot::handleStateReceivedCaptchaPromptUpdated() const {
  LOG() << "Got captcha. Sending answer\n";
  const auto captchaAnswerPacket = packet::building::ClientGatewayLoginIbuvAnswer::packet(worldState_.selfState().kCaptchaAnswer);
  packetBroker_.injectPacket(captchaAnswerPacket, PacketContainer::Direction::kClientToServer);
}

void Bot::handleStateCharacterListUpdated() const {
  LOG() << "Char list received: [ ";
  for (const auto &i : worldState_.selfState().characterList) {
    std::cout << i.name << ' ';
  }
  std::cout << "]\n";

  // Search for our character in the character list
  auto it = std::find_if(worldState_.selfState().characterList.begin(), worldState_.selfState().characterList.end(), [this](const packet::structures::CharacterSelection::Character &character) {
    return character.name == loginData_.name;
  });
  if (it == worldState_.selfState().characterList.end()) {
    LOG() << "Unable to find character \"" << loginData_.name << "\"\n";
    return;
  }

  // Found our character, select it
  auto charSelectionPacket = packet::building::ClientAgentCharacterSelectionJoinRequest::packet(loginData_.name);
  packetBroker_.injectPacket(charSelectionPacket, PacketContainer::Direction::kClientToServer);
}

// ============================================================================================================================
// ==================================================Movement event handling===================================================
// ============================================================================================================================

void Bot::handleEntityMovementBegan(const event::EntityMovementBegan &event) {
  entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(event.globalId);
  if (!mobileEntity.moving()) {
    throw std::runtime_error("Got an entity movement began event, but it is not moving");
  }
  const auto currentPosition = mobileEntity.position();
  if (mobileEntity.destinationPosition) {
    if (event.globalId == worldState_.selfState().globalId) {
    // TODO: We ought to combine these two UI functions
      userInterface_.broadcastMovementBeganUpdate(currentPosition, *worldState_.selfState().destinationPosition, worldState_.selfState().currentSpeed());
    } else {
      userInterface_.broadcastEntityMovementBegan(event.globalId, currentPosition, *mobileEntity.destinationPosition, mobileEntity.currentSpeed());
    }
  } else {
    if (event.globalId == worldState_.selfState().globalId) {
    // TODO: We ought to combine these two UI functions
      userInterface_.broadcastMovementBeganUpdate(currentPosition, worldState_.selfState().angle(), worldState_.selfState().currentSpeed());
    } else {
      userInterface_.broadcastEntityMovementBegan(event.globalId, currentPosition, mobileEntity.angle(), mobileEntity.currentSpeed());
    }
  }
  onUpdate(&event);
}

void Bot::handleEntityMovementEnded(const event::EntityMovementEnded &event) {
  entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(event.globalId);
  if (event.globalId == worldState_.selfState().globalId) {
    // TODO: We ought to combine these two UI functions
    userInterface_.broadcastMovementEndedUpdate(mobileEntity.position());
  } else {
    userInterface_.broadcastEntityMovementEnded(event.globalId, mobileEntity.position());
  }
  onUpdate(&event);
}

void Bot::handleEntityMovementTimerEnded(sro::scalar_types::EntityGlobalId globalId) {
  entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(globalId);
  mobileEntity.movementTimerCompleted(eventBroker_);
}

void Bot::handleEntityPositionUpdated(sro::scalar_types::EntityGlobalId globalId) {
  entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(globalId);
  if (mobileEntity.moving()) {
    throw std::runtime_error("Should never happen while moving");
  }

  // Not moving
  const auto currentPosition = mobileEntity.position();
  if (globalId == worldState_.selfState().globalId) {
    userInterface_.broadcastPositionChangedUpdate(currentPosition);
  } else {
    userInterface_.broadcastEntityPositionChanged(mobileEntity.globalId, currentPosition);
  }
}

void Bot::handleEntityNotMovingAngleChanged(sro::scalar_types::EntityGlobalId globalId) {
  if (globalId == worldState_.selfState().globalId) {
    // We only send the angle of the controlled character to the UI
    entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(globalId);
    userInterface_.broadcastNotMovingAngleChangedUpdate(mobileEntity.angle());
  }
}

void Bot::handleEntityEnteredGeometry(const event::EntityEnteredGeometry &event) {
  LOG() << "Entity " << event.globalId << " entered geometry" << std::endl;
  onUpdate();
}

void Bot::handleEntityExitedGeometry(const event::EntityExitedGeometry &event) {
  LOG() << "Entity " << event.globalId << " exited geometry" << std::endl;
  onUpdate();
}

// ============================================================================================================================
// ===============================================Character info packet handling===============================================
// ============================================================================================================================

void Bot::handleSpawned() {
  userInterface_.broadcastCharacterSpawn();
  const auto &currentLevelData = gameData_.levelData().getLevel(worldState_.selfState().getCurrentLevel());
  userInterface_.broadcastCharacterLevelUpdate(worldState_.selfState().getCurrentLevel(), currentLevelData.exp_C);
  userInterface_.broadcastCharacterExperienceUpdate(worldState_.selfState().getCurrentExperience(), worldState_.selfState().getCurrentSpExperience());
  userInterface_.broadcastCharacterSpUpdate(worldState_.selfState().getSkillPoints());
  userInterface_.broadcastCharacterNameUpdate(worldState_.selfState().name);
  userInterface_.broadcastGoldAmountUpdate(worldState_.selfState().getGold(), broadcast::ItemLocation::kCharacterInventory);
  const auto &regionName = gameData_.textZoneNameData().getRegionName(worldState_.selfState().position().regionId());
  userInterface_.broadcastMovementEndedUpdate(worldState_.selfState().position());
  LOG() << "Spawned at position " << worldState_.selfState().position() << std::endl;
  userInterface_.broadcastRegionNameUpdate(regionName);
  // Send entire inventory
  for (uint8_t inventorySlotIndex=0; inventorySlotIndex<worldState_.selfState().inventory.size(); ++inventorySlotIndex) {
    if (worldState_.selfState().inventory.hasItem(inventorySlotIndex)) {
      broadcastItemUpdateForSlot(broadcast::ItemLocation::kCharacterInventory, worldState_.selfState().inventory, inventorySlotIndex);
    }
  }
  // Send avatar inventory too
  for (uint8_t inventorySlotIndex=0; inventorySlotIndex<worldState_.selfState().avatarInventory.size(); ++inventorySlotIndex) {
    if (worldState_.selfState().avatarInventory.hasItem(inventorySlotIndex)) {
      broadcastItemUpdateForSlot(broadcast::ItemLocation::kAvatarInventory, worldState_.selfState().avatarInventory, inventorySlotIndex);
    }
  }
}

void Bot::handleCosSpawned(const event::CosSpawned &event) {
  // Send COS inventory
  auto it = worldState_.selfState().cosInventoryMap.find(event.cosGlobalId);
  if (it == worldState_.selfState().cosInventoryMap.end()) {
    throw std::runtime_error("Received COS Spawned event, but dont have this COS");
  }
  const auto &cosInventory = it->second;
  for (uint8_t inventorySlotIndex=0; inventorySlotIndex<cosInventory.size(); ++inventorySlotIndex) {
    if (cosInventory.hasItem(inventorySlotIndex)) {
      broadcastItemUpdateForSlot(broadcast::ItemLocation::kCosInventory, cosInventory, inventorySlotIndex);
    }
  }
}

void Bot::handleVitalsChanged() {
  if (!worldState_.selfState().maxHp() || !worldState_.selfState().maxMp()) {
    // Dont yet know our max
    return;
  }
  onUpdate();
}

void Bot::handleStatesChanged() {
  onUpdate();
}

// ============================================================================================================================
// ===========================================================Skills===========================================================
// ============================================================================================================================

void Bot::handleSkillBegan(const event::SkillBegan &event) {
  if (event.casterGlobalId == worldState_.selfState().globalId) {
    const auto skillName = gameData_.getSkillNameIfExists(event.skillRefId);
    LOG() << "Our skill \"" << (skillName ? *skillName : "UNKNOWN") << "\" began" << std::endl;
    onUpdate(&event);
  }
}

void Bot::handleSkillEnded(const event::SkillEnded &event) {
  if (event.casterGlobalId == worldState_.selfState().globalId) {
    const auto skillName = gameData_.getSkillNameIfExists(event.skillRefId);
    LOG() << "Our skill \"" << (skillName ? *skillName : "UNKNOWN") << "\" ended" << std::endl;
    onUpdate(&event);
  }
}

void Bot::handleSkillCooldownEnded(const event::SkillCooldownEnded &event) {
  const auto skillName = gameData_.getSkillNameIfExists(event.skillRefId);
  LOG() << "Skill " << event.skillRefId << "(" << (skillName ? *skillName : "UNKNOWN") << ") cooldown ended" << std::endl;
  worldState_.selfState().skillsOnCooldown.erase(event.skillRefId);
  onUpdate();
}

// ============================================================================================================================
// ============================================================Misc============================================================
// ============================================================================================================================

void Bot::storageInitialized() {
  for (uint8_t storageSlotIndex=0; storageSlotIndex<worldState_.selfState().storage.size(); ++storageSlotIndex) {
    if (worldState_.selfState().storage.hasItem(storageSlotIndex)) {
      broadcastItemUpdateForSlot(broadcast::ItemLocation::kStorage, worldState_.selfState().storage, storageSlotIndex);
    }
  }
  onUpdate();
}

void Bot::guildStorageInitialized() {
  for (uint8_t storageSlotIndex=0; storageSlotIndex<worldState_.selfState().guildStorage.size(); ++storageSlotIndex) {
    if (worldState_.selfState().guildStorage.hasItem(storageSlotIndex)) {
      broadcastItemUpdateForSlot(broadcast::ItemLocation::kGuildStorage, worldState_.selfState().guildStorage, storageSlotIndex);
    }
  }
}

void Bot::inventoryUpdated(const event::InventoryUpdated &inventoryUpdatedEvent) {
  if (inventoryUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kCharacterInventory, worldState_.selfState().inventory, *inventoryUpdatedEvent.srcSlotNum);
  }
  if (inventoryUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kCharacterInventory, worldState_.selfState().inventory, *inventoryUpdatedEvent.destSlotNum);
  }
  onUpdate(&inventoryUpdatedEvent);
}

void Bot::avatarInventoryUpdated(const event::AvatarInventoryUpdated &avatarInventoryUpdatedEvent) {
  if (avatarInventoryUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kAvatarInventory, worldState_.selfState().avatarInventory, *avatarInventoryUpdatedEvent.srcSlotNum);
  }
  if (avatarInventoryUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kAvatarInventory, worldState_.selfState().avatarInventory, *avatarInventoryUpdatedEvent.destSlotNum);
  }
}

void Bot::cosInventoryUpdated(const event::CosInventoryUpdated &cosInventoryUpdatedEvent) {
  auto it = worldState_.selfState().cosInventoryMap.find(cosInventoryUpdatedEvent.globalId);
  if (it == worldState_.selfState().cosInventoryMap.end()) {
    throw std::runtime_error("COS inventory updated, but not tracking this COS");
  }
  const auto &cosInventory = it->second;
  if (cosInventoryUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kCosInventory, cosInventory, *cosInventoryUpdatedEvent.srcSlotNum);
  }
  if (cosInventoryUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kCosInventory, cosInventory, *cosInventoryUpdatedEvent.destSlotNum);
  }
}

void Bot::storageUpdated(const event::StorageUpdated &storageUpdatedEvent) {
  if (storageUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kStorage, worldState_.selfState().storage, *storageUpdatedEvent.srcSlotNum);
  }
  if (storageUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kStorage, worldState_.selfState().storage, *storageUpdatedEvent.destSlotNum);
  }
  onUpdate(&storageUpdatedEvent);
}

void Bot::guildStorageUpdated(const event::GuildStorageUpdated &guildStorageUpdatedEvent) {
  if (guildStorageUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kGuildStorage, worldState_.selfState().guildStorage, *guildStorageUpdatedEvent.srcSlotNum);
  }
  if (guildStorageUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kGuildStorage, worldState_.selfState().guildStorage, *guildStorageUpdatedEvent.destSlotNum);
  }
}

void Bot::broadcastItemUpdateForSlot(broadcast::ItemLocation itemLocation, const storage::Storage &itemStorage, const uint8_t slotIndex) {
  uint16_t quantity{0};
  std::optional<std::string> itemName;
  if (itemStorage.hasItem(slotIndex)) {
    const auto *item = itemStorage.getItem(slotIndex);
    if (const auto *itemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(item)) {
      quantity = itemAsExpendable->quantity;
    } else {
      // Not an expendable, only 1
      quantity = 1;
    }
    itemName = gameData_.textItemAndSkillData().getItemName(item->itemInfo->nameStrID128);
  }
  userInterface_.broadcastItemUpdate(itemLocation, slotIndex, quantity, itemName);
}

void Bot::entitySpawned(const event::EntitySpawned &event) {
  const bool trackingEntity = worldState_.entityTracker().trackingEntity(event.globalId);
  if (!trackingEntity) {
    throw std::runtime_error("Received entity spawned event, but we're not tracking this entity");
  }
  const auto *entity = worldState_.entityTracker().getEntity(event.globalId);
  {
    // TODO: Remove. This is a temporary mechanism to measure the maximum visibility range.
    // According to Daxter:
    //  maximum possible should be around 905
    //  given that you can at most see almost 2 blocks across
    const auto distanceToEntity = sro::position_math::calculateDistance2d(worldState_.selfState().position(), entity->position());
    if (distanceToEntity > worldState_.selfState().estimatedVisibilityRange) {
      LOG() << "Bumping up estimated visibility range to " << distanceToEntity << std::endl;
      worldState_.selfState().estimatedVisibilityRange = distanceToEntity;
    }
  }
  userInterface_.broadcastEntitySpawned(entity);
  if (const auto *characterEntity = dynamic_cast<const entity::Character*>(entity)) {
    if (characterEntity->lifeState == sro::entity::LifeState::kDead) {
      // Entity spawned in as dead
      // TODO: Create a more comprehensive entity proto which contains life state
      userInterface_.broadcastEntityLifeStateChanged(characterEntity->globalId, characterEntity->lifeState);
    }
  }

  onUpdate(&event);
}

void Bot::entityDespawned(const event::EntityDespawned &event) {
  userInterface_.broadcastEntityDespawned(event.globalId);

  onUpdate(&event);
}

void Bot::entityLifeStateChanged(const event::EntityLifeStateChanged &event) {
  entity::Character &characterEntity = worldState_.getEntity<entity::Character>(event.globalId);
  userInterface_.broadcastEntityLifeStateChanged(characterEntity.globalId, characterEntity.lifeState);
  onUpdate();
}

void Bot::itemUseTimedOut(const event::ItemUseTimeout &event) {
  // TODO: Refactor this whole itemUsedTimeout concept
  worldState_.selfState().itemUsedTimeoutTimer.reset();
  worldState_.selfState().removedItemFromUsedItemQueue(event.slotNum, event.typeData);
  onUpdate();
}

void Bot::handleKnockbackStunEnded() {
  selfState().stunnedFromKnockback = false;
  onUpdate();
}

void Bot::handleKnockdownStunEnded() {
  selfState().stunnedFromKnockdown = false;
  onUpdate();
}

void Bot::handleItemCooldownEnded(const event::ItemCooldownEnded &event) {
  selfState().itemCooldownEnded(event.typeId);
  onUpdate(&event);
}
