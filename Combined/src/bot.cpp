#include "bot.hpp"
#include "helpers.hpp"
#include "logging.hpp"

#include "packet/building/clientAgentActionDeselectRequest.hpp"
#include "packet/building/clientAgentActionSelectRequest.hpp"
#include "packet/building/clientAgentActionTalkRequest.hpp"
#include "packet/building/clientAgentAuthRequest.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"
#include "packet/building/clientAgentCharacterSelectionJoinRequest.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"
#include "packet/building/clientAgentInventoryStorageOpenRequest.hpp"
#include "packet/building/clientGatewayLoginIbuvAnswer.hpp"
#include "packet/building/clientGatewayLoginRequest.hpp"

Bot::Bot(const config::CharacterLoginData &loginData,
         const pk2::GameData &gameData,
         Proxy &proxy,
         broker::PacketBroker &broker) :
      loginData_(loginData),
      gameData_(gameData),
      proxy_(proxy),
      broker_(broker) {
  eventBroker_.run();
  userInterface_.run();
  subscribeToEvents();
}

state::Self& Bot::selfState() {
  return selfState_;
}

Proxy& Bot::proxy() const {
  return proxy_;
}

broker::PacketBroker& Bot::packetBroker() const {
  return broker_;
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
  eventBroker_.subscribeToEvent(event::EventCode::kStateCharacterListUpdated, eventHandleFunction);
  // Movement events
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementBegan, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementTimerEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntitySyncedPosition, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEnteredNewRegion, eventHandleFunction);
  // Character info events
  eventBroker_.subscribeToEvent(event::EventCode::kSpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCosSpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kItemWaitForReuseDelay, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kHpPotionCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMpPotionCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kVigorPotionCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kUniversalPillCooldownEnded, eventHandleFunction);
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
  eventBroker_.subscribeToEvent(event::EventCode::kPurificationPillCooldownEnded, eventHandleFunction);
#endif
  eventBroker_.subscribeToEvent(event::EventCode::kHpChanged, eventHandleFunction);
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
}

void Bot::handleEvent(const event::Event *event) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);

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
      case event::EventCode::kStateCharacterListUpdated:
        handleStateCharacterListUpdated();
        break;

      // Movement events
      case event::EventCode::kEntityMovementBegan:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityMovementBegan&>(*event);
          handleEntityMovementBegan(castedEvent.globalId);
          break;
        }
      case event::EventCode::kEntityMovementEnded:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityMovementEnded&>(*event);
          handleEntityMovementEnded(castedEvent.globalId);
          break;
        }
      case event::EventCode::kEntityMovementTimerEnded:
        {
          const auto &castedEvent = dynamic_cast<const event::EntityMovementTimerEnded&>(*event);
          handleEntityMovementTimerEnded(castedEvent.globalId);
          break;
        }
      case event::EventCode::kEntitySyncedPosition:
        {
          const auto &castedEvent = dynamic_cast<const event::EntitySyncedPosition&>(*event);
          handleEntitySyncedPosition(castedEvent.globalId);
          break;
        }
      case event::EventCode::kEnteredNewRegion:
        {
          const auto pos = selfState_.position();
          const auto &regionName = gameData_.textZoneNameData().getRegionName(pos.regionId());
          userInterface_.broadcastRegionNameUpdate(regionName);
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
      case event::EventCode::kItemWaitForReuseDelay:
        {
          const event::ItemWaitForReuseDelay &castedEvent = dynamic_cast<const event::ItemWaitForReuseDelay&>(*event);
          handleItemWaitForReuseDelay(castedEvent);
        }
        break;
      case event::EventCode::kHpPotionCooldownEnded:
      case event::EventCode::kMpPotionCooldownEnded:
      case event::EventCode::kVigorPotionCooldownEnded:
        handlePotionCooldownEnded(eventCode);
        break;
      case event::EventCode::kUniversalPillCooldownEnded:
      case event::EventCode::kPurificationPillCooldownEnded:
        handlePillCooldownEnded(eventCode);
        break;
      case event::EventCode::kHpChanged:
        userInterface_.broadcastCharacterHpUpdate(selfState_.hp());
        handleVitalsChanged();
        break;
      case event::EventCode::kMpChanged:
        userInterface_.broadcastCharacterMpUpdate(selfState_.mp());
        handleVitalsChanged();
        break;
      case event::EventCode::kMaxHpMpChanged:
        if (selfState_.maxHp() && selfState_.maxMp()) {
          userInterface_.broadcastCharacterMaxHpMpUpdate(*selfState_.maxHp(), *selfState_.maxMp());
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
        userInterface_.broadcastGoldAmountUpdate(selfState_.getGold(), broadcast::ItemLocation::kCharacterInventory);
        break;
      case event::EventCode::kStorageGoldUpdated:
        userInterface_.broadcastGoldAmountUpdate(selfState_.getStorageGold(), broadcast::ItemLocation::kStorage);
        break;
      case event::EventCode::kGuildStorageGoldUpdated:
        userInterface_.broadcastGoldAmountUpdate(selfState_.getGuildStorageGold(), broadcast::ItemLocation::kGuildStorage);
        break;
      case event::EventCode::kCharacterSkillPointsUpdated:
        userInterface_.broadcastCharacterSpUpdate(selfState_.getSkillPoints());
        break;
      case event::EventCode::kCharacterExperienceUpdated:
        userInterface_.broadcastCharacterExperienceUpdate(selfState_.getCurrentExperience(), selfState_.getCurrentSpExperience());
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
  handleVitals();

  if (!selfState_.trainingIsActive) {
    // Not training, nothing else to do
    return;
  }

  // Note: Assuming we start in the spawnpoint of Jangan
  // Which is somewhere near position { 25000, 951.0f, -33.0f, 1372.0f }

  if (!stateMachine_) {
    throw std::runtime_error("We should have a state machine");
  }
  stateMachine_->onUpdate(event);
}

void Bot::handleVitals() {
  // TODO: Check if we're in a state where using items is possible
  checkIfNeedToUsePill();
  checkIfNeedToHeal();
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
  if (selfState_.trainingIsActive) {
    LOG() << "Asked to start training, but we're already training" << std::endl;
    return;
  }

  selfState_.trainingIsActive = true;
  // TODO: Should we stop whatever we're doing?
  //  For example, if we're running, stop where we are

  // Initialize state machine
  stateMachine_.emplace(*this);

  // Trigger onUpdate
  onUpdate();
}

void Bot::stopTraining() {
  // TODO: Need to cleanup current action to avoid leaving the client in a bad state
  //  Ex. Need to close a shop npc dialog
  selfState_.trainingIsActive = false;
  stateMachine_.reset();
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
  broker_.injectPacket(packet, direction);
}

// ============================================================================================================================
// ================================================Login process event handling================================================
// ============================================================================================================================

void Bot::handleStateShardIdUpdated() const {
  // We received the server list from the server, try to log in
  const auto loginAuthPacket = packet::building::ClientGatewayLoginRequest::packet(gameData_.divisionInfo().locale, loginData_.id, loginData_.password, selfState_.shardId);
  broker_.injectPacket(loginAuthPacket, PacketContainer::Direction::kClientToServer);
}

void Bot::handleStateConnectedToAgentServerUpdated() {
  const auto clientAuthPacket = packet::building::ClientAgentAuthRequest::packet(selfState_.token, loginData_.id, loginData_.password, gameData_.divisionInfo().locale, selfState_.kMacAddress);
  broker_.injectPacket(clientAuthPacket, PacketContainer::Direction::kClientToServer);
  // Set our state to logging in so that we'll know to block packets from the client if it tries to also login
  selfState_.loggingIn = true;
}

void Bot::handleStateReceivedCaptchaPromptUpdated() const {
  LOG() << "Got captcha. Sending answer\n";
  const auto captchaAnswerPacket = packet::building::ClientGatewayLoginIbuvAnswer::packet(selfState_.kCaptchaAnswer);
  broker_.injectPacket(captchaAnswerPacket, PacketContainer::Direction::kClientToServer);
}

void Bot::handleStateCharacterListUpdated() const {
  LOG() << "Char list received: [ ";
  for (const auto &i : selfState_.characterList) {
    std::cout << i.name << ' ';
  }
  std::cout << "]\n";

  // Search for our character in the character list
  auto it = std::find_if(selfState_.characterList.begin(), selfState_.characterList.end(), [this](const packet::structures::CharacterSelection::Character &character) {
    return character.name == loginData_.name;
  });
  if (it == selfState_.characterList.end()) {
    LOG() << "Unable to find character \"" << loginData_.name << "\"\n";
    return;
  }

  // Found our character, select it
  auto charSelectionPacket = packet::building::ClientAgentCharacterSelectionJoinRequest::packet(loginData_.name);
  broker_.injectPacket(charSelectionPacket, PacketContainer::Direction::kClientToServer);
}

// ============================================================================================================================
// ==================================================Movement event handling===================================================
// ============================================================================================================================

void Bot::handleEntityMovementBegan(sro::scalar_types::EntityGlobalId globalId) {
  entity::MobileEntity &mobileEntity = getMobileEntity(globalId);
  if (!mobileEntity.moving) {
    throw std::runtime_error("Got an entity movement began event, but it is not moving");
  }
  const auto currentPosition = mobileEntity.position();
  if (mobileEntity.destinationPosition) {
    if (globalId == selfState_.globalId) {
      userInterface_.broadcastMovementBeganUpdate(currentPosition, *selfState_.destinationPosition, selfState_.currentSpeed());
    } else {
      userInterface_.broadcastEntityMovementBegan(globalId, currentPosition, *mobileEntity.destinationPosition, mobileEntity.currentSpeed());
    }
  } else {
    if (globalId == selfState_.globalId) {
      userInterface_.broadcastMovementBeganUpdate(currentPosition, *selfState_.movementAngle, selfState_.currentSpeed());
    } else {
      userInterface_.broadcastEntityMovementBegan(globalId, currentPosition, *mobileEntity.movementAngle, mobileEntity.currentSpeed());
    }
  }
}

void Bot::handleEntityMovementEnded(sro::scalar_types::EntityGlobalId globalId) {
  entity::MobileEntity &mobileEntity = getMobileEntity(globalId);
  bool needToRunUpdate{false};
  if (globalId == selfState_.globalId) {
    // TODO: We ought to combine these two UI functions
    userInterface_.broadcastMovementEndedUpdate(mobileEntity.position());
    needToRunUpdate = true;
  } else {
    userInterface_.broadcastEntityMovementEnded(globalId, mobileEntity.position());
  }

  if (needToRunUpdate) {
    onUpdate();
  }
}

void Bot::handleEntityMovementTimerEnded(sro::scalar_types::EntityGlobalId globalId) {
  entity::MobileEntity &mobileEntity = getMobileEntity(globalId);
  mobileEntity.movementTimerCompleted(eventBroker_);
}

void Bot::handleEntitySyncedPosition(sro::scalar_types::EntityGlobalId globalId) {
  entity::MobileEntity &mobileEntity = getMobileEntity(globalId);
  const auto currentPosition = mobileEntity.position();
  if (mobileEntity.moving) {
    if (mobileEntity.destinationPosition) {
      if (globalId == selfState_.globalId) {
        // TODO: We ought to combine these two UI functions
        userInterface_.broadcastMovementBeganUpdate(currentPosition, *mobileEntity.destinationPosition, mobileEntity.currentSpeed());
      } else {
        userInterface_.broadcastEntityMovementBegan(mobileEntity.globalId, currentPosition, *mobileEntity.destinationPosition, mobileEntity.currentSpeed());
      }
    } else {
      if (globalId == selfState_.globalId) {
        // TODO: We ought to combine these two UI functions
        userInterface_.broadcastMovementBeganUpdate(currentPosition, *mobileEntity.movementAngle, mobileEntity.currentSpeed());
      } else {
        userInterface_.broadcastEntityMovementBegan(mobileEntity.globalId, currentPosition, *mobileEntity.movementAngle, mobileEntity.currentSpeed());
      }
    }
  } else {
    // Not moving
    if (globalId == selfState_.globalId) {
      // TODO: !!!
      LOG() << "Whoa" << std::endl;
    } else {
      userInterface_.broadcastEntityPositionChanged(mobileEntity.globalId, currentPosition);
    }
  }
}

// ============================================================================================================================
// ===============================================Character info packet handling===============================================
// ============================================================================================================================

void Bot::handleSpawned() {
  userInterface_.broadcastCharacterSpawn();
  const auto &currentLevelData = gameData_.levelData().getLevel(selfState_.getCurrentLevel());
  userInterface_.broadcastCharacterLevelUpdate(selfState_.getCurrentLevel(), currentLevelData.exp_C);
  userInterface_.broadcastCharacterExperienceUpdate(selfState_.getCurrentExperience(), selfState_.getCurrentSpExperience());
  userInterface_.broadcastCharacterSpUpdate(selfState_.getSkillPoints());
  userInterface_.broadcastCharacterNameUpdate(selfState_.name);
  userInterface_.broadcastGoldAmountUpdate(selfState_.getGold(), broadcast::ItemLocation::kCharacterInventory);
  const auto &regionName = gameData_.textZoneNameData().getRegionName(selfState_.position().regionId());
  userInterface_.broadcastMovementEndedUpdate(selfState_.position());
  userInterface_.broadcastRegionNameUpdate(regionName);
  // Send entire inventory
  for (uint8_t inventorySlotIndex=0; inventorySlotIndex<selfState_.inventory.size(); ++inventorySlotIndex) {
    if (selfState_.inventory.hasItem(inventorySlotIndex)) {
      broadcastItemUpdateForSlot(broadcast::ItemLocation::kCharacterInventory, selfState_.inventory, inventorySlotIndex);
    }
  }
  // Send avatar inventory too
  for (uint8_t inventorySlotIndex=0; inventorySlotIndex<selfState_.avatarInventory.size(); ++inventorySlotIndex) {
    if (selfState_.avatarInventory.hasItem(inventorySlotIndex)) {
      broadcastItemUpdateForSlot(broadcast::ItemLocation::kAvatarInventory, selfState_.avatarInventory, inventorySlotIndex);
    }
  }
}

void Bot::handleCosSpawned(const event::CosSpawned &event) {
  // Send COS inventory
  auto it = selfState_.cosInventoryMap.find(event.cosGlobalId);
  if (it == selfState_.cosInventoryMap.end()) {
    throw std::runtime_error("Received COS Spawned event, but dont have this COS");
  }
  const auto &cosInventory = it->second;
  for (uint8_t inventorySlotIndex=0; inventorySlotIndex<cosInventory.size(); ++inventorySlotIndex) {
    if (cosInventory.hasItem(inventorySlotIndex)) {
      broadcastItemUpdateForSlot(broadcast::ItemLocation::kCosInventory, cosInventory, inventorySlotIndex);
    }
  }
}

void Bot::handleItemWaitForReuseDelay(const event::ItemWaitForReuseDelay &event) {
  LOG() << "Failed to use ";
  if (event.itemTypeId == helpers::type_id::makeTypeId(3, 3, 1, 1)) {
    std::cout << "hp";
  } else if (event.itemTypeId == helpers::type_id::makeTypeId(3, 3, 1, 2)) {
    std::cout << "mp";
  } else if (event.itemTypeId == helpers::type_id::makeTypeId(3, 3, 1, 3)) {
    std::cout << "vigor";
  }
  std::cout << " potion because there's still a cooldown, going to retry" << std::endl;
  useItem(event.inventorySlotNum, event.itemTypeId);
}

void Bot::handlePotionCooldownEnded(const event::EventCode eventCode) {
  LOG() << "Potion cooldown ended" << std::endl;
  if (eventCode == event::EventCode::kHpPotionCooldownEnded) {
    selfState_.resetHpPotionEventId();
  } else if (eventCode == event::EventCode::kMpPotionCooldownEnded) {
    selfState_.resetMpPotionEventId();
  } else if (eventCode == event::EventCode::kVigorPotionCooldownEnded) {
    selfState_.resetVigorPotionEventId();
  } else {
    LOG() << "Unhandled potion cooldown ended event\n";
  }
  onUpdate();
}

void Bot::handlePillCooldownEnded(const event::EventCode eventCode) {
  LOG() << "Pill cooldown ended" << std::endl;
  if (eventCode == event::EventCode::kUniversalPillCooldownEnded) {
    selfState_.resetUniversalPillEventId();
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
  } else if (eventCode == event::EventCode::kPurificationPillCooldownEnded) {
    selfState_.resetPurificationPillEventId();
#endif
  } else {
    LOG() << "Unhandled pill cooldown ended event\n";
  }
  onUpdate();
}

void Bot::handleVitalsChanged() {
  if (!selfState_.maxHp() || !selfState_.maxMp()) {
    // Dont yet know our max
    return;
  }
  onUpdate();
}

void Bot::handleStatesChanged() {
  onUpdate();
}

// ============================================================================================================================
// ============================================================Misc============================================================
// ============================================================================================================================

// ============================================================================================================================
// ====================================================Actual action logic=====================================================
// ============================================================================================================================

void Bot::checkIfNeedToHeal() {
  if (!selfState_.maxHp() || !selfState_.maxMp()) {
    // Dont yet know our max
    LOG() << "checkIfNeedToHeal: dont know max hp or mp\n";
    return;
  }
  if (*selfState_.maxHp() == 0) {
    // Dead, cant heal
    // TODO: Get from state update instead
    LOG() << "checkIfNeedToHeal: Dead, cant heal\n";
    return;
  }
  const double hpPercentage = static_cast<double>(selfState_.hp())/(*selfState_.maxHp());
  const double mpPercentage = static_cast<double>(selfState_.mp())/(*selfState_.maxMp());

  const auto legacyStateEffects = selfState_.legacyStateEffects();
  const bool haveZombie = (legacyStateEffects[helpers::toBitNum(packet::enums::AbnormalStateFlag::kZombie)] > 0);

  // TODO: Investigate if using multiple potions in one go causes issues
  if (!alreadyUsedPotion(PotionType::kVigor)) {
    if (!haveZombie && (hpPercentage < kVigorThreshold_ || mpPercentage < kVigorThreshold_)) {
      usePotion(PotionType::kVigor);
    }
  }
  if (!alreadyUsedPotion(PotionType::kHp)) {
    if (!haveZombie && hpPercentage < kHpThreshold_) {
      usePotion(PotionType::kHp);
    }
  }
  if (!alreadyUsedPotion(PotionType::kMp)) {
    if (mpPercentage < kMpThreshold_) {
      usePotion(PotionType::kMp);
    }
  }
}

bool Bot::alreadyUsedPotion(PotionType potionType) {
  if (potionType == PotionType::kHp) {
    if (selfState_.haveHpPotionEventId()) {
      return true;
    }
    const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 1, 1);
    return selfState_.itemIsInUsedItemQueue(itemTypeId);
  } else if (potionType == PotionType::kMp) {
    if (selfState_.haveMpPotionEventId()) {
      return true;
    }
    const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 1, 2);
    return selfState_.itemIsInUsedItemQueue(itemTypeId);
  } else if (potionType == PotionType::kVigor) {
    if (selfState_.haveVigorPotionEventId()) {
      return true;
    }
    const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 1, 3);
    return selfState_.itemIsInUsedItemQueue(itemTypeId);
  }
  // TODO: Handle other cases
  return false;
}

void Bot::usePotion(PotionType potionType) {
  // We enter this funciton assuming that:
  //  1. The potion isnt on cooldown
  //  2. We have the potion

  uint8_t typeId4;
  if (potionType == PotionType::kHp) {
    typeId4 = 1;
  } else if (potionType == PotionType::kMp) {
    typeId4 = 2;
  } else if (potionType == PotionType::kVigor) {
    typeId4 = 3;
  } else {
    std::cout << "CharacterInfoModule::usePotion: Potion type " << static_cast<int>(potionType) << " not supported\n";
    return;
  }

  // Find potion in inventory
  for (uint8_t slotNum=0; slotNum<selfState_.inventory.size(); ++slotNum) {
    if (!selfState_.inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = selfState_.inventory.getItem(slotNum);
    const storage::ItemExpendable *item;
    if ((item = dynamic_cast<const storage::ItemExpendable*>(itemPtr)) != nullptr) {
      // Expendable item
      if (item->itemInfo->typeId1 == 3 && item->itemInfo->typeId2 == 3 && item->itemInfo->typeId3 == 1 && item->itemInfo->typeId4 == typeId4) {
        if (typeId4 == 3 || item->itemInfo->param2 == 0 && item->itemInfo->param4 == 0) {
          // Avoid hp/mp grains
          useItem(slotNum, itemPtr->typeData());
          return;
        }
      }
    }
  }
  // Dont have the item we were looking for
}

void Bot::checkIfNeedToUsePill() {
  const auto legacyStateEffects = selfState_.legacyStateEffects();
  if (std::any_of(legacyStateEffects.begin(), legacyStateEffects.end(), [](const uint16_t effect){ return effect > 0; })) {
    // Need to use a universal pill
    if (!alreadyUsedUniversalPill()) {
      useUniversalPill();
    }
  }
  const auto modernStateLevels = selfState_.modernStateLevels();
  if (std::any_of(modernStateLevels.begin(), modernStateLevels.end(), [](const uint8_t level){ return level > 0; })) {
    // Need to use purification pill
    if (!alreadyUsedPurificationPill()) {
      usePurificationPill();
    }
  }
}

bool Bot::alreadyUsedUniversalPill() {
  if (selfState_.haveUniversalPillEventId()) {
    return true;
  }
  // Pill isnt on cooldown, but maybe we already queued a use of it
  const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 2, 6);
  return selfState_.itemIsInUsedItemQueue(itemTypeId);
}

bool Bot::alreadyUsedPurificationPill() {
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
  if (selfState_.havePurificationPillEventId()) {
    return true;
  }
#endif
  const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 2, 1);
  return selfState_.itemIsInUsedItemQueue(itemTypeId);
}

void Bot::useUniversalPill() {
  // Figure out our status with the highest effect
  const auto legacyStateEffects = selfState_.legacyStateEffects();
  uint16_t ourWorstStatusEffect = *std::max_element(legacyStateEffects.begin(), legacyStateEffects.end());
  int32_t bestCure = 0;
  uint8_t bestOptionSlotNum;
  uint16_t bestOptionTypeData;

  for (uint8_t slotNum=0; slotNum<selfState_.inventory.size(); ++slotNum) {
    if (!selfState_.inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = selfState_.inventory.getItem(slotNum);
    const storage::ItemExpendable *item;
    if ((item = dynamic_cast<const storage::ItemExpendable*>(itemPtr)) != nullptr) {
      // Expendable item
      if (item->itemInfo->typeId1 == 3 && item->itemInfo->typeId2 == 3 && item->itemInfo->typeId3 == 2 && item->itemInfo->typeId4 == 6) {
        // Universal pill
        if (bestCure == 0) {
          // First pill found, at least we can use this
          bestCure = item->itemInfo->param1;
          bestOptionSlotNum = slotNum;
          bestOptionTypeData = itemPtr->typeData();
        } else {
          // Already have a choice, lets see if this is better
          const auto thisPillCureEffect = item->itemInfo->param1;
          const bool curesEverything = (thisPillCureEffect >= ourWorstStatusEffect);
          const bool curesMoreThanPrevious = (thisPillCureEffect >= ourWorstStatusEffect && bestCure < ourWorstStatusEffect);
          if (curesEverything && thisPillCureEffect < bestCure) {
            // Found a smaller pill that can cure everything
            bestCure = thisPillCureEffect;
            bestOptionSlotNum = slotNum;
            bestOptionTypeData = itemPtr->typeData();
          } else if (curesMoreThanPrevious && thisPillCureEffect > bestCure) {
            // Found a pill that can cure more without being wasteful
            bestCure = thisPillCureEffect;
            bestOptionSlotNum = slotNum;
            bestOptionTypeData = itemPtr->typeData();
          }
        }
      }
    }
  }
  if (bestCure != 0) {
    // Found the best pill
    useItem(bestOptionSlotNum, bestOptionTypeData);
  }
}

void Bot::usePurificationPill() {
  const auto modernStateLevels = selfState_.modernStateLevels();
  int32_t currentCureLevel = 0;
  uint8_t bestOptionSlotNum;
  uint16_t bestOptionTypeData;

  for (uint8_t slotNum=0; slotNum<selfState_.inventory.size(); ++slotNum) {
    if (!selfState_.inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = selfState_.inventory.getItem(slotNum);
    const storage::ItemExpendable *item;
    if ((item = dynamic_cast<const storage::ItemExpendable*>(itemPtr)) != nullptr) {
      // Expendable item
      if (item->itemInfo->typeId1 == 3 && item->itemInfo->typeId2 == 3 && item->itemInfo->typeId3 == 2 && item->itemInfo->typeId4 == 1) {
        // Purification pill
        const auto pillCureStateBitmask = item->itemInfo->param1;
        const auto curableStatesWeHave = (pillCureStateBitmask & selfState_.stateBitmask());
        if (curableStatesWeHave > 0) {
          // This pill will cure at least some of the type of state(s) that we have
          const auto pillTreatmentLevel = item->itemInfo->param2;
          if (pillTreatmentLevel != currentCureLevel) {
            std::vector<uint8_t> stateLevels;
            for (uint32_t bitNum=0; bitNum<32; ++bitNum) {
              const auto bit = 1 << bitNum;
              if (curableStatesWeHave & bit) {
                stateLevels.push_back(modernStateLevels[bitNum]);
              }
            }
            const bool curesEverything = (*std::max_element(stateLevels.begin(), stateLevels.end()) <= pillTreatmentLevel);
            const bool curesMoreThanPrevious = (std::find_if(stateLevels.begin(), stateLevels.end(), [&pillTreatmentLevel, &currentCureLevel](const uint8_t lvl){
              return ((lvl > currentCureLevel) && (lvl <= pillTreatmentLevel));
            }) != stateLevels.end());

            if (pillTreatmentLevel < currentCureLevel && curesEverything) {
              // Found a smaller pill that is completely sufficient
              currentCureLevel = pillTreatmentLevel;
              bestOptionSlotNum = slotNum;
              bestOptionTypeData = itemPtr->typeData();
            } else if (pillTreatmentLevel > currentCureLevel && curesMoreThanPrevious) {
              // Found a bigger pill that does more than the previous
              currentCureLevel = pillTreatmentLevel;
              bestOptionSlotNum = slotNum;
              bestOptionTypeData = itemPtr->typeData();
            }
          }
        }
      }
    }
  }
  if (currentCureLevel != 0) {
    // Found the best pill
    useItem(bestOptionSlotNum, bestOptionTypeData);
  }
}

void Bot::useItem(uint8_t slotNum, uint16_t typeData) {
  uint8_t typeId1 = (typeData >> 2) & 0b111;
  uint8_t typeId2 = (typeData >> 5) & 0b11;
  uint8_t typeId3 = (typeData >> 7) & 0b1111;
  uint8_t typeId4 = (typeData >> 11) & 0b11111;
  if (typeId1 == 3 && typeId2 == 3 && typeId3 == 1) {
    // Potion
    if (typeId4 == 1) {
      if (alreadyUsedPotion(PotionType::kHp)) {
        // Already used an Hp potion, not going to re-queue
        return;
      }
    } else if (typeId4 == 2) {
      if (alreadyUsedPotion(PotionType::kMp)) {
        // Already used an Mp potion, not going to re-queue
        return;
      }
    } else if (typeId4 == 3) {
      if (alreadyUsedPotion(PotionType::kVigor)) {
        // Already used a Vigor potion, not going to re-queue
        return;
      }
    }
  }
  broker_.injectPacket(packet::building::ClientAgentInventoryItemUseRequest::packet(slotNum, typeData), PacketContainer::Direction::kClientToServer);
  selfState_.pushItemToUsedItemQueue(slotNum, typeData);
}

void Bot::storageInitialized() {
  for (uint8_t storageSlotIndex=0; storageSlotIndex<selfState_.storage.size(); ++storageSlotIndex) {
    if (selfState_.storage.hasItem(storageSlotIndex)) {
      broadcastItemUpdateForSlot(broadcast::ItemLocation::kStorage, selfState_.storage, storageSlotIndex);
    }
  }
  onUpdate();
}

void Bot::guildStorageInitialized() {
  for (uint8_t storageSlotIndex=0; storageSlotIndex<selfState_.guildStorage.size(); ++storageSlotIndex) {
    if (selfState_.guildStorage.hasItem(storageSlotIndex)) {
      broadcastItemUpdateForSlot(broadcast::ItemLocation::kGuildStorage, selfState_.guildStorage, storageSlotIndex);
    }
  }
}

void Bot::inventoryUpdated(const event::InventoryUpdated &inventoryUpdatedEvent) {
  if (inventoryUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kCharacterInventory, selfState_.inventory, *inventoryUpdatedEvent.srcSlotNum);
  }
  if (inventoryUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kCharacterInventory, selfState_.inventory, *inventoryUpdatedEvent.destSlotNum);
  }
  onUpdate(&inventoryUpdatedEvent);
}

void Bot::avatarInventoryUpdated(const event::AvatarInventoryUpdated &avatarInventoryUpdatedEvent) {
  if (avatarInventoryUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kAvatarInventory, selfState_.avatarInventory, *avatarInventoryUpdatedEvent.srcSlotNum);
  }
  if (avatarInventoryUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kAvatarInventory, selfState_.avatarInventory, *avatarInventoryUpdatedEvent.destSlotNum);
  }
}

void Bot::cosInventoryUpdated(const event::CosInventoryUpdated &cosInventoryUpdatedEvent) {
  auto it = selfState_.cosInventoryMap.find(cosInventoryUpdatedEvent.globalId);
  if (it == selfState_.cosInventoryMap.end()) {
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
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kStorage, selfState_.storage, *storageUpdatedEvent.srcSlotNum);
  }
  if (storageUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kStorage, selfState_.storage, *storageUpdatedEvent.destSlotNum);
  }
  onUpdate(&storageUpdatedEvent);
}

void Bot::guildStorageUpdated(const event::GuildStorageUpdated &guildStorageUpdatedEvent) {
  if (guildStorageUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kGuildStorage, selfState_.guildStorage, *guildStorageUpdatedEvent.srcSlotNum);
  }
  if (guildStorageUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(broadcast::ItemLocation::kGuildStorage, selfState_.guildStorage, *guildStorageUpdatedEvent.destSlotNum);
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
  const bool trackingEntity = entityTracker_.trackingEntity(event.globalId);
  if (!trackingEntity) {
    throw std::runtime_error("Received entity spawned event, but we're not tracking this entity");
  }
  const auto *entity = entityTracker_.getEntity(event.globalId);
  userInterface_.broadcastEntitySpawned(event.globalId, entity->position(), entity->entityType());
}

void Bot::entityDespawned(const event::EntityDespawned &event) {
  userInterface_.broadcastEntityDespawned(event.globalId);
}

// ============================================================================================================================
// ==========================================================Helpers===========================================================
// ============================================================================================================================

entity::MobileEntity& Bot::getMobileEntity(sro::scalar_types::EntityGlobalId globalId) {
  if (globalId == selfState_.globalId) {
    return selfState_;
  } else if (entityTracker_.trackingEntity(globalId)) {
    return entityTracker_.getEntity<entity::MobileEntity>(globalId);
  } else {
    throw std::runtime_error("Trying to get untracked mobile entity");
  }
}