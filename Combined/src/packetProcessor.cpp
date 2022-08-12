#include "helpers.hpp"
#include "logging.hpp"
#include "packetProcessor.hpp"

#include "packet/opcode.hpp"

#define ENFORCE_PURIFICATION_PILL_COOLDOWN

#define TRY_CAST_AND_HANDLE_PACKET(PACKET_TYPE, HANDLE_FUNCTION_NAME) \
{ \
  auto *castedParsedPacket = dynamic_cast<PACKET_TYPE*>(parsedPacket.get()); \
  if (castedParsedPacket != nullptr) { \
    return HANDLE_FUNCTION_NAME(*castedParsedPacket); \
  } \
}

PacketProcessor::PacketProcessor(state::Entity &entityState,
                                 state::Self &selfState,
                                 broker::PacketBroker &brokerSystem,
                                 broker::EventBroker &eventBroker,
                                 ui::UserInterface &userInterface,
                                 const packet::parsing::PacketParser &packetParser,
                                 const pk2::GameData &gameData) :
      entityState_(entityState),
      selfState_(selfState),
      broker_(brokerSystem),
      eventBroker_(eventBroker),
      userInterface_(userInterface),
      packetParser_(packetParser),
      gameData_(gameData) {
  subscribeToPackets();
}

void PacketProcessor::subscribeToPackets() {
  auto packetHandleFunction = std::bind(&PacketProcessor::handlePacket, this, std::placeholders::_1);

  // Server packets
  //   Login packets
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentAuthRequest, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::LOGIN_SERVER_LIST, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::LOGIN_SERVER_AUTH_INFO, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::LOGIN_CLIENT_INFO, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_LOGIN_RESULT, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_CHARACTER, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_INGAME_ACCEPT, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerGatewayLoginIbuvChallenge, packetHandleFunction);
  // broker_.subscribeToServerPacket(static_cast<packet::Opcode>(0x6005), packetHandleFunction);
  //   Movement packets
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateMovement, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdatePosition, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntitySyncPosition, packetHandleFunction);
  //   Character info packets
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentInventoryOperationRequest, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterData, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryStorageData, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateState, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateStatus, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentAbnormalInfo, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterUpdateStats, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryItemUseResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryOperationResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateMoveSpeed, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityGroupspawnData, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntitySpawn, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityDespawn, packetHandleFunction);

  //   Misc. packets
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionDeselectResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionSelectResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionTalkResponse, packetHandleFunction);
  // broker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionDeselectRequest, packetHandleFunction);
  // broker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionSelectRequest, packetHandleFunction);
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionTalkRequest, packetHandleFunction);
}

bool PacketProcessor::handlePacket(const PacketContainer &packet) const {
  std::unique_ptr<packet::parsing::ParsedPacket> parsedPacket;
  try {
    parsedPacket = packetParser_.parsePacket(packet);
  } catch (std::exception &ex) {
    std::cerr << "[PacketProcessor] Failed to parse packet " << std::hex << packet.opcode << std::dec << "\n  Error: \"" << ex.what() << "\"\n";
    return true;
  }

  if (!parsedPacket) {
    // Not yet parsing this packet
    return true;
  }

  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);

  try {
    // Login packet handlers
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedLoginServerList, serverListReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedLoginResponse, loginResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedLoginClientInfo, loginClientInfoReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedUnknown, unknownPacketReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAuthResponse, serverAuthReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentCharacterSelectionActionResponse, charListReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse, charSelectionJoinResponseReceived);

    // Movement packet handlers
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateMovement, serverAgentEntityUpdateMovementReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdatePosition, serverAgentEntityUpdatePositionReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntitySyncPosition, serverAgentEntitySyncPositionReceived);

    // Character info packet handlers
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedClientItemMove, clientItemMoveReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentCharacterData, serverAgentCharacterDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentInvetoryStorageData, serverAgentInventoryStorageDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateState, serverAgentEntityUpdateStateReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateMoveSpeed, serverAgentEntityUpdateMoveSpeedReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentEntityUpdateStatus, serverAgentEntityUpdateStatusReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentAbnormalInfo, serverAgentAbnormalInfoReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentCharacterUpdateStats, serverAgentCharacterUpdateStatsReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentInventoryItemUseResponse, serverAgentInventoryItemUseResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryOperationResponse, serverAgentInventoryOperationResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentEntityGroupSpawnData, serverAgentEntityGroupSpawnDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentSpawn, serverAgentSpawnReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentDespawn, serverAgentDespawnReceived);

    // Misc. packets
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentActionDeselectResponse, serverAgentDeselectResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentActionSelectResponse, serverAgentSelectResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentActionTalkResponse, serverAgentTalkResponseReceived);
    // TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentActionDeselectRequest, clientAgentActionDeselectRequestReceived);
    // TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentActionSelectRequest, clientAgentActionSelectRequestReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentActionTalkRequest, clientAgentActionTalkRequestReceived);

  } catch (std::exception &ex) {
    LOG() << "Error while handling packet!\n  " << ex.what() << std::endl;
  }

  LOG() << "Unhandled packet subscribed to\n";
  return true;
}

// ============================================================================================================================
// ===============================================Login process packet handling================================================
// ============================================================================================================================

bool PacketProcessor::serverListReceived(const packet::parsing::ParsedLoginServerList &packet) const {
  selfState_.shardId = packet.shardId();
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateShardIdUpdated));
  return true;
}

bool PacketProcessor::loginResponseReceived(const packet::parsing::ParsedLoginResponse &packet) const {
  if (packet.result() == packet::enums::LoginResult::kSuccess) {
    selfState_.token = packet.token();
  } else {
    LOG() << " Login failed\n";
  }
  return true;
}

bool PacketProcessor::loginClientInfoReceived(const packet::parsing::ParsedLoginClientInfo &packet) const {
  // This packet is a response to the client sending 0x2001 where the client indicates that it is the "SR_Client"
  if (packet.serviceName() == "AgentServer") {
    // Connected to agentserver, send client auth packet
    selfState_.connectedToAgentServer = true;
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateConnectedToAgentServerUpdated));
  }
  return true;
}

bool PacketProcessor::unknownPacketReceived(const packet::parsing::ParsedUnknown &packet) const {
  if (packet.opcode() == packet::Opcode::kClientAgentAuthRequest) {
    // The client is trying to authenticate
    if (selfState_.loggingIn) {
      // We're the ones logging in, the client is not in the correct state, so this packet will break the login process
      // Block this from the server
      return false;
    }
  } else if (packet.opcode() == packet::Opcode::kServerGatewayLoginIbuvChallenge) {
    // Got the captcha prompt, respond with an answer
    selfState_.receivedCaptchaPrompt = true;
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateReceivedCaptchaPromptUpdated));
  }
  return true;
}

bool PacketProcessor::serverAuthReceived(const packet::parsing::ParsedServerAuthResponse &packet) const {
  if (packet.result() == 0x01) {
    // Successful login
    selfState_.loggingIn = false;
    // Client will automatically request the character listing
    // TODO: For clientless, we will need to do this ourself
  }
  return true;
}

bool PacketProcessor::charListReceived(const packet::parsing::ParsedServerAgentCharacterSelectionActionResponse &packet) const {
  selfState_.characterList = packet.characters();
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateCharacterListUpdated));
  return true;
}

bool PacketProcessor::charSelectionJoinResponseReceived(const packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse &packet) const {
  // A character was selected after login, this is the response
  if (packet.result() != 0x01) {
    // Character selection failed
    // TODO: Properly handle error
    LOG() << "Failed when selecting character\n";
  }
  return true;
}

// ============================================================================================================================
// ==================================================Movement packet handling==================================================
// ============================================================================================================================

bool PacketProcessor::serverAgentEntitySyncPositionReceived(packet::parsing::ServerAgentEntitySyncPosition &packet) const {
  if (packet.globalId() == selfState_.globalId()) {
    const auto &currentPosition = selfState_.position();
    LOG() << "Syncing position: " << currentPosition.xOffset << ',' << currentPosition.zOffset << std::endl;
    selfState_.syncPosition(packet.position());
  }
  return false;
}

bool PacketProcessor::serverAgentEntityUpdatePositionReceived(packet::parsing::ServerAgentEntityUpdatePosition &packet) const {
  if (packet.globalId() == selfState_.globalId()) {
    if (selfState_.moving()) {
      LOG() << "Position update received" << std::endl;
      // Happens when you collide with something
      // Note: this also happens when running to pick an item
      // Note: I think this also happens when a speed drug is cancelled
      if (selfState_.haveMovingEventId()) {
        LOG() << "Cancelling movement timer\n";
        eventBroker_.cancelDelayedEvent(selfState_.getMovingEventId());
        selfState_.resetMovingEventId();
      }
    } else {
      LOG() << "We werent moving, weird\n";
      const auto pos = selfState_.position();
      LOG() << "Expected pos: " << pos.xOffset << ',' << pos.zOffset << '\n';
      LOG() << "Received pos: " << packet.position().xOffset << ',' << packet.position().zOffset << '\n';
      // TODO: Does it make sense to update our position in this case? Probably
      //  But it also seems like a problem because we mistakenly thought we were moving
    }
    selfState_.setPosition(packet.position());
    LOG() << "Now stationary at " << selfState_.position() << '\n';
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMovementEnded));
  }
  return true;
}

bool PacketProcessor::serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet) const {
  if (packet.globalId() == selfState_.globalId()) {
    packet::structures::Position sourcePosition;
    if (packet.hasSource()) {
      // Server is telling us our source position
      sourcePosition = packet.sourcePosition();
      const auto currentPosition = selfState_.position();
      if (std::round(currentPosition.xOffset) != std::round(sourcePosition.xOffset) && std::round(currentPosition.zOffset) != std::round(sourcePosition.zOffset)) {
        // We arent where we thought we were
        // We need to cancel this movement. Either move back to the source position or try to stop exactly where we are (by moving to our estimated position)
        LOG() << "Whoa, we're a bit off from where we thought we were. Expected: " << currentPosition.xOffset << ',' << currentPosition.zOffset << ", actual: " << sourcePosition.xOffset << ',' << sourcePosition.zOffset << std::endl;
      }
      LOG() << "Syncing src position: " << sourcePosition.xOffset << ',' << sourcePosition.zOffset << std::endl;
      selfState_.syncPosition(sourcePosition);
    } else {
      // Server doesnt tell us where we're coming from, use our internally tracked position
      sourcePosition = selfState_.position();
    }
    if (selfState_.haveMovingEventId()) {
      // Had a timer already running for movement, cancel it
      LOG() << "Had a running timer, cancelling it" << std::endl;
      eventBroker_.cancelDelayedEvent(selfState_.getMovingEventId());
      selfState_.resetMovingEventId();
    }
    LOG() << "We are moving from " << sourcePosition << ' ';
    if (packet.hasDestination()) {
      auto destPosition = packet.destinationPosition();
      std::cout << "to " << destPosition << '\n';
      if (sourcePosition.xOffset == destPosition.xOffset && sourcePosition.zOffset == destPosition.zOffset) {
        LOG() << "Server says we're moving to our current position. wtf?\n";
        // Ignore this
      } else {
        auto seconds = helpers::secondsToTravel(sourcePosition, destPosition, selfState_.currentSpeed());
        LOG() << "Should take " << seconds << "s. Timer set\n";
        const auto movingEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMovementTimerEnded), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
        selfState_.setMovingEventId(movingEventId);
        selfState_.setMoving(packet.destinationPosition());
      }
    } else {
      std::cout << "toward " << packet.angle() << '\n';
      selfState_.setMoving(packet.angle());
    }
  }
  return true;
}

// ============================================================================================================================
// ===============================================Character info packet handling===============================================
// ============================================================================================================================

bool PacketProcessor::clientItemMoveReceived(const packet::parsing::ParsedClientItemMove &packet) const {
  const auto itemMovement = packet.movement();
  if (itemMovement.type == packet::enums::ItemMovementType::kBuyFromNPC) {
    // User is buying something from the store
    selfState_.setUserPurchaseRequest(itemMovement);
  }
  return true;
}

bool PacketProcessor::serverAgentCharacterDataReceived(const packet::parsing::ParsedServerAgentCharacterData &packet) const {
  selfState_.initialize(packet.entityUniqueId(), packet.refObjId(), packet.hp(), packet.mp(), packet.masteries(), packet.skills());

  // Position
  selfState_.setPosition(packet.position());
  LOG() << "Our Ref Obj Id " << packet.refObjId() << '\n';
  LOG() << "Position: " << (packet.position().isDungeon() ? "dungeon " : "world ");
  if (packet.position().isDungeon()) {
    std::cout << '#' << (int)packet.position().dungeonId();
  } else {
    std::cout << "region (" << (int)packet.position().xSector() << ',' << (int)packet.position().zSector() << ")";
  }
  std::cout << " (" << packet.position().xOffset << ',' << packet.position().yOffset << ',' << packet.position().zOffset << ")\n";
  LOG() << "{" << packet.position().regionId << ',' << packet.position().xOffset << "f," << packet.position().yOffset << "f," << packet.position().zOffset << "f}" << std::endl;

  // State
  selfState_.setLifeState(packet.lifeState());
  selfState_.setMotionState(packet.motionState());
  selfState_.setBodyState(packet.bodyState());

  // Speed
  selfState_.setSpeed(packet.walkSpeed(), packet.runSpeed());
  selfState_.setHwanSpeed(packet.hwanSpeed());
  auto refObjId = packet.refObjId();
  selfState_.setGold(packet.gold());
  LOG() << "Gold: " << selfState_.getGold() << std::endl;
  selfState_.setRaceAndGender(refObjId);
  const auto inventorySize = packet.inventorySize();
  const auto &inventoryItemMap = packet.inventoryItemMap();
  helpers::initializeInventory(selfState_.inventory, inventorySize, inventoryItemMap);
  LOG() << "Inventory initialized\n";

  LOG() << "GID:" << selfState_.globalId() << ", and we have " << selfState_.hp() << " hp and " << selfState_.mp() << " mp\n";
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kSpawned));
  return true;
}

bool PacketProcessor::serverAgentInventoryStorageDataReceived(const packet::parsing::ParsedServerAgentInvetoryStorageData &packet) const {
  selfState_.storageGold_ = packet.gold();
  helpers::initializeInventory(selfState_.storage, packet.storageSize(), packet.storageItemMap());
  LOG() << "Storage initialized\n";
  selfState_.haveOpenedStorageSinceTeleport = true;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStorageOpened));
  return true;
}

bool PacketProcessor::serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet) const {
  if (selfState_.spawned() && packet.gId() == selfState_.globalId()) {
    if (packet.stateType() == packet::parsing::StateType::kBodyState) {
      selfState_.setBodyState(static_cast<packet::enums::BodyState>(packet.state()));
    } else if (packet.stateType() == packet::parsing::StateType::kLifeState) {
      selfState_.setLifeState(static_cast<packet::enums::LifeState>(packet.state()));
      if (static_cast<packet::enums::LifeState>(packet.state()) == packet::enums::LifeState::kDead) {
        LOG() << "CharacterInfoModule: We died, clearing used item queue" << std::endl;
        selfState_.clearUsedItemQueue();
      }
    } else if (packet.stateType() == packet::parsing::StateType::kMotionState) {
      if (static_cast<packet::enums::MotionState>(packet.state()) == packet::enums::MotionState::kWalk) {
        LOG() << "Motion state update to walk\n";
      } else if (static_cast<packet::enums::MotionState>(packet.state()) == packet::enums::MotionState::kRun) {
        LOG() << "Motion state update to run\n";
      } else if (static_cast<packet::enums::MotionState>(packet.state()) == packet::enums::MotionState::kSit) {
        LOG() << "Motion state update to sit\n";
      } else {
        LOG() << "Motion state update to " << static_cast<int>(packet.state()) << '\n';
      }
      selfState_.setMotionState(static_cast<packet::enums::MotionState>(packet.state()));
    }
  }
  return true;
}

bool PacketProcessor::serverAgentEntityUpdateMoveSpeedReceived(const packet::parsing::ServerAgentEntityUpdateMoveSpeed &packet) const {
  if (packet.globalId() == selfState_.globalId()) {
    // Our speed was updated
    LOG() << "Our speed was updated from " << selfState_.walkSpeed() << ',' << selfState_.runSpeed() << " to " << packet.walkSpeed() << ',' << packet.runSpeed() << '\n';
    selfState_.setSpeed(packet.walkSpeed(), packet.runSpeed());
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kCharacterSpeedUpdated));
  }
  return true;
}

bool PacketProcessor::serverAgentEntityUpdateStatusReceived(const packet::parsing::ParsedServerAgentEntityUpdateStatus &packet) const {
  if (packet.entityUniqueId() != selfState_.globalId()) {
    // Not for my character, can ignore
    return true;
  }
  if (packet.vitalBitmask() & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoHp)) {
    // Our HP changed
    if (selfState_.hp() != packet.newHpValue()) {
      selfState_.setHp(packet.newHpValue());
      eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kHpPercentChanged));
    } else {
      LOG() << "Weird, says HP changed, but it didn't\n";
    }
  }
  if (packet.vitalBitmask() & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoMp)) {
    // Our MP changed
    if (selfState_.mp() != packet.newMpValue()) {
      selfState_.setMp(packet.newMpValue());
      eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMpPercentChanged));
    } else {
      LOG() << "Weird, says MP changed, but it didn't\n";
    }
  }

  if (packet.vitalBitmask() & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoAbnormal)) {
    // Our states changed
    auto stateBitmask = packet.stateBitmask();
    auto stateLevels = packet.stateLevels();
    selfState_.updateStates(stateBitmask, stateLevels);
    LOG() << "Updated our states\n";
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStatesChanged));
  }
  return true;
}

bool PacketProcessor::serverAgentAbnormalInfoReceived(const packet::parsing::ParsedServerAgentAbnormalInfo &packet) const {
  for (int i=0; i<=helpers::toBitNum(packet::enums::AbnormalStateFlag::kZombie); ++i) {
    selfState_.setLegacyStateEffect(helpers::fromBitNum(i), packet.states()[i].effectOrLevel);
  }
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStatesChanged));
  return true;
}

bool PacketProcessor::serverAgentCharacterUpdateStatsReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet) const {
  selfState_.setMaxHpMp(packet.maxHp(), packet.maxMp());
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kHpPercentChanged));
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMpPercentChanged));
  return true;
}

bool PacketProcessor::serverAgentInventoryItemUseResponseReceived(const packet::parsing::ParsedServerAgentInventoryItemUseResponse &packet) const {
  if (packet.result() == 1) {
    // Successfully used an item
    if (selfState_.inventory.hasItem(packet.slotNum())) {
      auto *itemPtr = selfState_.inventory.getItem(packet.slotNum());
      // Lets double check it's type data
      if (packet.itemData() == itemPtr->typeData()) {
        auto *expendableItemPtr = dynamic_cast<storage::ItemExpendable*>(itemPtr);
        if (expendableItemPtr != nullptr) {
          expendableItemPtr->quantity = packet.remainingCount();
          if (helpers::type_id::isHpPotion(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a potion
            if (selfState_.haveHpPotionEventId()) {
              LOG() << "Uhhhh, supposedly successfully used an hp potion when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(selfState_.getHpPotionEventId());
            }
            LOG() << "Successfully used a hpPotion\n";
            const auto hpPotionDelay = selfState_.getHpPotionDelay() + kPotionDelayBufferMs_;
            const auto hpPotionEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kHpPotionCooldownEnded), std::chrono::milliseconds(hpPotionDelay));
            selfState_.setHpPotionEventId(hpPotionEventId);
          } else if (helpers::type_id::isMpPotion(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a potion
            if (selfState_.haveMpPotionEventId()) {
              LOG() << "Uhhhh, supposedly successfully used an mp potion when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(selfState_.getMpPotionEventId());
            }
            LOG() << "Successfully used a mpPotion\n";
            const auto mpPotionDelay = selfState_.getMpPotionDelay() + kPotionDelayBufferMs_;
            const auto mpPotionEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMpPotionCooldownEnded), std::chrono::milliseconds(mpPotionDelay));
            selfState_.setMpPotionEventId(mpPotionEventId);
          } else if (helpers::type_id::isVigorPotion(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a potion
            if (selfState_.haveVigorPotionEventId()) {
              LOG() << "Uhhhh, supposedly successfully used a vigor potion when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(selfState_.getVigorPotionEventId());
            }
            LOG() << "Successfully used a vigorPotion\n";
            // TODO: Grains and regular potions have different delays, at least for Eu chars
            const auto vigorPotionDelay = selfState_.getVigorPotionDelay() + kPotionDelayBufferMs_;
            const auto vigorPotionEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kVigorPotionCooldownEnded), std::chrono::milliseconds(vigorPotionDelay));
            selfState_.setVigorPotionEventId(vigorPotionEventId);
          } else if (helpers::type_id::isUniversalPill(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a pill
            if (selfState_.haveUniversalPillEventId()) {
              LOG() << "Uhhhh, supposedly successfully used a universal pill when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(selfState_.getUniversalPillEventId());
            }
            const auto universalPillEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kUniversalPillCooldownEnded), std::chrono::milliseconds(selfState_.getUniversalPillDelay()));
            selfState_.setUniversalPillEventId(universalPillEventId);
          } else if (helpers::type_id::isPurificationPill(*expendableItemPtr->itemInfo)) {
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
            // Set a timeout for how long we must wait before retrying to use a pill
            if (selfState_.havePurificationPillEventId()) {
              LOG() << "Uhhhh, supposedly successfully used a purification pill when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(selfState_.getPurificationPillEventId());
            }
            const auto purificationPillEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kPurificationPillCooldownEnded), std::chrono::milliseconds(selfState_.getPurificationPillDelay()));
            selfState_.setPurificationPillEventId(purificationPillEventId);
#endif
          }
          if (expendableItemPtr->quantity == 0) {
            LOG() << "Used the last of this item! Delete from inventory\n";
            // TODO: Instead, delete the item upon receiving server_item_movement in the case DEL_ITEM_BY_SERVER
            selfState_.inventory.deleteItem(packet.slotNum());
          }
        }
      }
    }
  } else {
    // Failed to use item
    if (!selfState_.usedItemQueueIsEmpty()) {
      // This was an item that we tried to use
      if (packet.errorCode() == packet::enums::InventoryErrorCode::kWaitForReuseDelay) {
        // TODO: When we start tracking items moving in the invetory, we'll need to somehow update this used item queue
        const auto usedItem = selfState_.getUsedItemQueueFront();
        eventBroker_.publishEvent(std::make_unique<event::ItemWaitForReuseDelay>(usedItem.inventorySlotNum, usedItem.itemTypeId));
      } else if (packet.errorCode() == packet::enums::InventoryErrorCode::kCharacterDead) {
        LOG() << "Failed to use item because we're dead\n";
      } else {
        LOG() << "Unknown error while trying to use an item: " << static_cast<int>(packet.errorCode()) << '\n';
      }
    }
  }
  selfState_.popItemFromUsedItemQueueIfNotEmpty();
  return true;
}

bool PacketProcessor::serverAgentInventoryOperationResponseReceived(const packet::parsing::ServerAgentInventoryOperationResponse &packet) const {
  // TODO: If we used an item and it moved, we'll need to update the "reference" to this item in the used item queue
  const std::vector<packet::structures::ItemMovement> &itemMovements = packet.itemMovements();
  for (const auto &movement : itemMovements) {
    if (movement.type == packet::enums::ItemMovementType::kWithinInventory) {
      selfState_.inventory.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      //TODO: Add event in other places
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kWithinStorage) {
      // Not handling because we dont parse the storage init packet
      selfState_.storage.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent(std::make_unique<event::StorageUpdated>(movement.srcSlot, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kInventoryToStorage) {
      auto item = selfState_.inventory.withdrawItem(movement.srcSlot);
      selfState_.storage.addItem(movement.destSlot, std::move(item));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::StorageUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kStorageToInventory) {
      auto item = selfState_.storage.withdrawItem(movement.srcSlot);
      selfState_.inventory.addItem(movement.destSlot, std::move(item));
      eventBroker_.publishEvent(std::make_unique<event::StorageUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kBuyFromNPC) {
      if (selfState_.haveUserPurchaseRequest()) {
        const auto userPurchaseRequest = selfState_.getUserPurchaseRequest();
        // User purchased something, we saved this so that we can get the NPC's global Id
        if (entityState_.trackingEntity(userPurchaseRequest.globalId)) {
          auto object = entityState_.getEntity(userPurchaseRequest.globalId);
          // Found the NPC which this purchase was made with
          if (gameData_.characterData().haveCharacterWithId(object->refObjId)) {
            auto npcName = gameData_.characterData().getCharacterById(object->refObjId).codeName128;
            auto itemInfo = gameData_.shopData().getItemFromNpc(npcName, userPurchaseRequest.storeTabNumber, userPurchaseRequest.storeSlotNumber);
            const auto &itemRef = gameData_.itemData().getItemByCodeName128(itemInfo.refItemCodeName);
            LOG() << "Bought " << movement.quantity << " x \"" << itemInfo.refItemCodeName << "\"(refId=" << itemRef.id << ") from \"" << npcName << "\"\n";
            if (movement.destSlots.size() == 1) {
              // Just a single item or single stack
              auto item = helpers::createItemFromScrap(itemInfo, itemRef);
              storage::ItemExpendable *itemExp = dynamic_cast<storage::ItemExpendable*>(item.get());
              if (itemExp != nullptr) {
                itemExp->quantity = movement.quantity;
              }
              selfState_.inventory.addItem(movement.destSlots[0], item);
              helpers::printItem(movement.destSlots[0], item.get(), gameData_);
              std::cout << '\n';
              eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlots[0]));
            } else {
              // Multiple destination slots, must be unstackable items like equipment
              for (auto destSlot : movement.destSlots) {
                auto item = helpers::createItemFromScrap(itemInfo, itemRef);
                selfState_.inventory.addItem(destSlot, item);
                helpers::printItem(movement.destSlot, item.get(), gameData_);
                std::cout << '\n';
                eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
              }
            }
          }
        }
        selfState_.resetUserPurchaseRequest();
      } else {
        LOG() << "kBuyFromNPC but we dont have the data from the client packet\n";
        // TODO: Introduce unknown item concept?
      }
    } else if (movement.type == packet::enums::ItemMovementType::kSellToNPC) {
      if (selfState_.inventory.hasItem(movement.srcSlot)) {
        bool soldEntireStack = true;
        auto item = selfState_.inventory.getItem(movement.srcSlot);
        storage::ItemExpendable *itemExpendable;
        if ((itemExpendable = dynamic_cast<storage::ItemExpendable*>(item)) != nullptr) {
          if (itemExpendable->quantity != movement.quantity) {
            LOG() << "Sold only some of this item " << itemExpendable->quantity << " -> " << itemExpendable->quantity-movement.quantity << '\n';
            soldEntireStack = false;
            itemExpendable->quantity -= movement.quantity;
            auto clonedItem = storage::cloneItem(item);
            dynamic_cast<storage::ItemExpendable*>(clonedItem.get())->quantity = movement.quantity;
            selfState_.buybackQueue.addItem(clonedItem);
          }
        }
        if (soldEntireStack) {
          auto item = selfState_.inventory.withdrawItem(movement.srcSlot);
          selfState_.buybackQueue.addItem(item);
        }
      } else {
        LOG() << "Sold an item from a slot that we didnt have item data for\n";
      }
    } else if (movement.type == packet::enums::ItemMovementType::kBuyback) {
      if (selfState_.buybackQueue.hasItem(movement.srcSlot)) {
        if (!selfState_.inventory.hasItem(movement.destSlot)) {
          const auto itemPtr = selfState_.buybackQueue.getItem(movement.srcSlot);
          // TODO: Track gold change
          //  The amount of gold that this item costs to buyback seems to be equal to the amount that it was sold for
          bool boughtBackAll = true;
          if (movement.quantity > 1) {
            storage::ItemExpendable *itemExpendable = dynamic_cast<storage::ItemExpendable*>(itemPtr);
            if (itemExpendable != nullptr) {
              if (itemExpendable->quantity > movement.quantity) {
                LOG() << "Only buying back a partial amount from the buyback slot. Didnt know this was possible (" << movement.quantity << '/' << itemExpendable->quantity << ")\n";
                boughtBackAll = false;
                auto clonedItem = storage::cloneItem(itemPtr);
                itemExpendable->quantity -= movement.quantity;
                dynamic_cast<storage::ItemExpendable*>(clonedItem.get())->quantity = movement.quantity;
                selfState_.inventory.addItem(movement.destSlot, clonedItem);
                LOG() << "Added item to inventory\n";
                helpers::printItem(movement.destSlot, clonedItem.get(), gameData_);
                std::cout << '\n';
              }
            }
          }
          if (boughtBackAll) {
            auto item = selfState_.buybackQueue.withdrawItem(movement.srcSlot);
            selfState_.inventory.addItem(movement.destSlot, item);
            LOG() << "Bought back entire stack\n";
            helpers::printItem(movement.destSlot, item.get(), gameData_);
          }
        } else {
          LOG() << "Bought back item is being moved into a slot that's already occupied\n";
        }
      } else {
        LOG() << "Bought back an item that we werent tracking\n";
      }
      LOG() << "Current buyback queue:\n";
      for (uint8_t slotNum=0; slotNum<selfState_.buybackQueue.size(); ++slotNum) {
        helpers::printItem(slotNum, selfState_.buybackQueue.getItem(slotNum), gameData_);
      }
      std::cout << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kPickItem) {
      if (movement.destSlot == packet::structures::ItemMovement::kGoldSlot) {
        LOG() << "Picked " << movement.goldPickAmount << " gold\n";
        selfState_.addGold(movement.goldPickAmount);
        LOG() << "Gold: " << selfState_.getGold() << std::endl;
        std::cout << '\n';
      } else {
        if (movement.pickedItem != nullptr) {
          LOG() << "Picked an item\n";
          if (selfState_.inventory.hasItem(movement.destSlot)) {
            LOG() << "Already something here\n";
            auto existingItem = selfState_.inventory.getItem(movement.destSlot);
            bool addedToStack = false;
            if (existingItem->refItemId == movement.pickedItem->refItemId) {
              // Both items have the same refId
              storage::ItemExpendable *newExpendableItem, *existingExpendableItem;
              if ((newExpendableItem = dynamic_cast<storage::ItemExpendable*>(movement.pickedItem.get())) &&
                  (existingExpendableItem = dynamic_cast<storage::ItemExpendable*>(existingItem))) {
                // Both items are expendables, so we can stack them
                // Picked item's quantity (if an expendable) is the total in the given slot
                existingExpendableItem->quantity = newExpendableItem->quantity;
                addedToStack = true;
              }
            }
            if (addedToStack) {
              LOG() << "Item added to stack\n";
            } else {
              LOG() << "Error: Item couldnt be added to the stack\n";
            }
          } else {
            LOG() << "New item!\n";
            selfState_.inventory.addItem(movement.destSlot, movement.pickedItem);
            LOG() << "Item " << (selfState_.inventory.hasItem(movement.destSlot) ? "was " : "was not ") << "successfully added\n";
          }
          helpers::printItem(movement.destSlot, movement.pickedItem.get(), gameData_);
        } else {
          LOG() << "Error: Picked an item, but the pickedItem is a nullptr\n";
        }
      }
      // This would be a good time to try to use a pill, potion, return scroll, etc.
    } else if (movement.type == packet::enums::ItemMovementType::kDropItem) {
      LOG() << "Dropped an item\n";
      if (selfState_.inventory.hasItem(movement.srcSlot)) {
        LOG() << "Dropping ";
        auto itemPtr = selfState_.inventory.withdrawItem(movement.srcSlot);
        helpers::printItem(movement.srcSlot, itemPtr.get(), gameData_);
        LOG() << "Item " << (!selfState_.inventory.hasItem(movement.srcSlot) ? "was " : "was not ") << "successfully dropped\n";
      } else {
        LOG() << "Error: But there's no item in this inventory slot\n";
      }
    } else if (movement.type == packet::enums::ItemMovementType::kGoldDrop) {
      selfState_.subtractGold(movement.goldAmount);
      LOG() << "Dropped " << movement.goldAmount << " gold\n";
      LOG() << "Gold: " << selfState_.getGold() << std::endl;
      LOG() << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kGoldStorageWithdraw) {
      selfState_.addGold(movement.goldAmount);
      LOG() << "Withdrew " << movement.goldAmount << " gold from storage\n";
      LOG() << "Gold: " << selfState_.getGold() << std::endl;
      LOG() << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kGoldStorageDeposit) {
      selfState_.subtractGold(movement.goldAmount);
      LOG() << "Deposited " << movement.goldAmount << " gold into storage\n";
      LOG() << "Gold: " << selfState_.getGold() << std::endl;
      LOG() << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kGoldGuildStorageDeposit) {
      selfState_.subtractGold(movement.goldAmount);
      LOG() << "Deposited " << movement.goldAmount << " gold into guild storage\n";
      LOG() << "Gold: " << selfState_.getGold() << std::endl;
      LOG() << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kGoldGuildStorageWithdraw) {
      selfState_.addGold(movement.goldAmount);
      LOG() << "Withdrew " << movement.goldAmount << " gold from guild storage\n";
      LOG() << "Gold: " << selfState_.getGold() << std::endl;
      LOG() << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kCosPickGold) {
      selfState_.addGold(movement.goldPickAmount);
      LOG() << "Pickpet picked " << movement.goldPickAmount << " gold\n";
      LOG() << "Gold: " << selfState_.getGold() << std::endl;
      LOG() << '\n';
    } else {
      LOG() << "Unknown item movement type: " << static_cast<int>(movement.type) << std::endl;
    }
  }
  return true;
}

bool PacketProcessor::serverAgentEntityGroupSpawnDataReceived(const packet::parsing::ParsedServerAgentEntityGroupSpawnData &packet) const {
  if (packet.groupSpawnType() == packet::parsing::GroupSpawnType::kSpawn) {
    for (auto obj : packet.objects()) {
      helpers::trackObject(entityState_, obj);
    }
  } else {
    for (auto gId : packet.despawns()) {
      helpers::stopTrackingObject(entityState_, gId);
    }
  }
  return true;
}

bool PacketProcessor::serverAgentSpawnReceived(const packet::parsing::ParsedServerAgentSpawn &packet) const {
  if (packet.object()) {
    helpers::trackObject(entityState_, packet.object());
  } else {
    LOG() << "Object spawned which we cannot track\n";
  }
  return true;
}

bool PacketProcessor::serverAgentDespawnReceived(const packet::parsing::ParsedServerAgentDespawn &packet) const {
  LOG() << "Going to stop tracking object with id " << packet.gId() << std::endl;
  helpers::stopTrackingObject(entityState_, packet.gId());
  return true;
}

// ============================================================================================================================
// ============================================================Misc============================================================
// ============================================================================================================================

// When the client requests to Talk, an Id is sent
// When the server responds that Talking is successful, no Id is returned
// To stop Talking, the client must send deselect with the Id

bool PacketProcessor::serverAgentDeselectResponseReceived(const packet::parsing::ServerAgentActionDeselectResponse &packet) const {
  if (packet.result() == 1) {
    // Successfully deselected
    // If there is a talk dialog, and we have an npc selected, it will take 2 deselects to close both dialogs
    //  First, the talk dialog is closed
    if (selfState_.talkingGidAndOption) {
      LOG() << "We're talking to an NPC, this closes the talk dialog" << std::endl;
      selfState_.talkingGidAndOption.reset();
      eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kEntityDeselected));
    } else {
      LOG() << "We were not talking to an NPC, maybe we have some entity selected" << std::endl;
      //  The entity is deselected
      if (selfState_.selectedEntity) {
        LOG() << "Deselecting " << *selfState_.selectedEntity << std::endl;
        selfState_.selectedEntity.reset();
        eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kEntityDeselected));
      } else {
        LOG() << "Weird, we didnt have anything selected\n";
      }
    }
  } else {
    LOG() << "Deselection failed" << std::endl;
  }
  return true;
}

bool PacketProcessor::serverAgentSelectResponseReceived(const packet::parsing::ServerAgentActionSelectResponse &packet) const {
  if (packet.result() == 1) {
    // Successfully selected
    LOG() << "We have successfully selected " << packet.gId() << std::endl;
    if (selfState_.selectedEntity) {
      // This happens if something is selected and then we select something else
      //  i.e. there is no deselect in between selects
      LOG() << "Weird, we already have something selected\n";
      // TODO: Maybe this is ok, maybe a deselect isnt required when switching between two entities
    }
    selfState_.selectedEntity = packet.gId();
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kEntitySelected));
    // ====================================================================================================
    // selfState_.selectedEntity = packet.gId();
  } else {
    LOG() << "Selection failed" << std::endl;
  }
  return true;
}

bool PacketProcessor::serverAgentTalkResponseReceived(const packet::parsing::ServerAgentActionTalkResponse &packet) const {
  if (packet.result() == 1) {
    LOG() << "We are now successfully talking to some npc" << std::endl;
    if (selfState_.pendingTalkGid) {
      // We were waiting for this response
      LOG() << "The npc we are talking to is gid " << *selfState_.pendingTalkGid << std::endl;
      selfState_.talkingGidAndOption = std::make_pair(*selfState_.pendingTalkGid, packet.talkOption());
      selfState_.pendingTalkGid.reset();
      // if (packet.talkOption() == packet::enums::TalkOption::kStorage) {
      //   // In the weird case of storage, the talk options dialog is automatically closed by the client
      //   LOG() << "In the weird case of storage, the talk options dialog is automatically closed by the client\n";
      //   selfState_.selectedEntity.reset();
      // }
      eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kNpcTalkStart));
    } else {
      LOG() << "Weird, we werent expecting to be talking to anything. As a result, we dont know what we're talking to" << std::endl;
    }
  } else {
    LOG() << "Failed to talk to NPC" << std::endl;
  }
  return true;
}

// bool PacketProcessor::clientAgentActionDeselectRequestReceived(const packet::parsing::ClientAgentActionDeselectRequest &packet) const {
//   return true;
// }

// bool PacketProcessor::clientAgentActionSelectRequestReceived(const packet::parsing::ClientAgentActionSelectRequest &packet) const {
//   return true;
// }

bool PacketProcessor::clientAgentActionTalkRequestReceived(const packet::parsing::ClientAgentActionTalkRequest &packet) const {
  LOG() << "Client is requesting to talk to " << packet.gId() << std::endl;
  if (selfState_.pendingTalkGid) {
    LOG() << "Weird, we're already waiting on a response from the server to talk to someone\n";
  } else {
    LOG() << "Setting that we're waiting on a response from the server to talk to someone\n";
    selfState_.pendingTalkGid = packet.gId();
  }
  return true;
}