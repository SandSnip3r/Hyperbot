#include "packetProcessor.hpp"

#include "entity/entity.hpp"
#include "helpers.hpp"
#include "logging.hpp"
#include "packet/opcode.hpp"

#include <silkroad_lib/position.h>

#define TRY_CAST_AND_HANDLE_PACKET(PACKET_TYPE, HANDLE_FUNCTION_NAME) \
{ \
  auto *castedParsedPacket = dynamic_cast<PACKET_TYPE*>(parsedPacket.get()); \
  if (castedParsedPacket != nullptr) { \
    return HANDLE_FUNCTION_NAME(*castedParsedPacket); \
  } \
}

PacketProcessor::PacketProcessor(state::EntityTracker &entityTracker,
                                 state::Self &selfState,
                                 broker::PacketBroker &brokerSystem,
                                 broker::EventBroker &eventBroker,
                                 ui::UserInterface &userInterface,
                                 const packet::parsing::PacketParser &packetParser,
                                 const pk2::GameData &gameData) :
      entityTracker_(entityTracker),
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
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateAngle, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateMovement, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdatePosition, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntitySyncPosition, packetHandleFunction);
  //   Character info packets
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentInventoryOperationRequest, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterData, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentCosData, packetHandleFunction);
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
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryRepairResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryUpdateDurability, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryUpdateItem, packetHandleFunction);
  // broker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionDeselectRequest, packetHandleFunction);
  // broker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionSelectRequest, packetHandleFunction);
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionTalkRequest, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdatePoints, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateExperience, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentGuildStorageData, packetHandleFunction);

  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionCommandResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillBegin, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillEnd, packetHandleFunction);
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
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateAngle, serverAgentEntityUpdateAngleReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateMovement, serverAgentEntityUpdateMovementReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdatePosition, serverAgentEntityUpdatePositionReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntitySyncPosition, serverAgentEntitySyncPositionReceived);

    // Character info packet handlers
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedClientItemMove, clientItemMoveReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentCharacterData, serverAgentCharacterDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentCosData, serverAgentCosDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentInventoryStorageData, serverAgentInventoryStorageDataReceived);
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
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryRepairResponse, serverAgentInventoryRepairResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryUpdateDurability, serverAgentInventoryUpdateDurabilityReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryUpdateItem, serverAgentInventoryUpdateItemReceived);
    // TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentActionDeselectRequest, clientAgentActionDeselectRequestReceived);
    // TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentActionSelectRequest, clientAgentActionSelectRequestReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentActionTalkRequest, clientAgentActionTalkRequestReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdatePoints, serverAgentEntityUpdatePointsReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateExperience, serverAgentEntityUpdateExperienceReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentGuildStorageData, serverAgentGuildStorageDataReceived);

    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentActionCommandResponse, serverAgentActionCommandResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentSkillBegin, serverAgentSkillBeginReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentSkillEnd, serverAgentSkillEndReceived);
  } catch (std::exception &ex) {
    LOG() << "Error while handling packet!\n  " << ex.what() << std::endl;
  }

  LOG() << "Unhandled packet subscribed to\n";
  return true;
}

void PacketProcessor::resetDataBecauseCharacterSpawned() const {
  // On teleport, COS will have different globaIds
  selfState_.cosInventoryMap.clear();
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

bool PacketProcessor::serverAgentEntityUpdateAngleReceived(packet::parsing::ServerAgentEntityUpdateAngle &packet) const {
  if (packet.globalId() == selfState_.globalId()) {
    LOG() << "Updated our angle" << std::endl;
    if (selfState_.moving()) {
      if (selfState_.haveDestination()) {
        throw std::runtime_error("Got angle update, but we're running to a destination");
      }
      if (selfState_.movementAngle() != packet.angle()) {
        // Changed angle while running
        selfState_.setMovingTowardAngle(std::nullopt, packet.angle());
      }
    }
  } else {
    if (!entityTracker_.trackingEntity(packet.globalId())) {
      throw std::runtime_error("Received movement update for something we're not tracking");
    }
    auto &mobileEntity = entityTracker_.getEntity<entity::MobileEntity>(packet.globalId());
    if (mobileEntity.moving) {
      if (mobileEntity.destinationPosition) {
        throw std::runtime_error("Got angle update, but we're running to a destination");
      }
      if (*mobileEntity.movementAngle != packet.angle()) {
        // Changed angle while running
        mobileEntity.setMovingTowardAngle(std::nullopt, packet.angle(), eventBroker_);
      }
    }
  }
  return true;
}

bool PacketProcessor::serverAgentEntitySyncPositionReceived(packet::parsing::ServerAgentEntitySyncPosition &packet) const {
  if (packet.globalId() == selfState_.globalId()) {
    // Sync our position to the new one
    LOG() << "Sync our position to " << packet.position() << std::endl;
    selfState_.syncPosition(packet.position());
  } else {
    auto &mobileEntity = entityTracker_.getEntity<entity::MobileEntity>(packet.globalId());
    mobileEntity.syncPosition(packet.position(), eventBroker_);
  }
  return true;
}

bool PacketProcessor::serverAgentEntityUpdatePositionReceived(packet::parsing::ServerAgentEntityUpdatePosition &packet) const {
  if (packet.globalId() == selfState_.globalId()) {
    LOG() << "Update our position to " << packet.position() << std::endl;
    selfState_.setStationaryAtPosition(packet.position());
  } else {
    auto &mobileEntity = entityTracker_.getEntity<entity::MobileEntity>(packet.globalId());
    mobileEntity.setStationaryAtPosition(packet.position(), eventBroker_);
  }
  return true;
}

bool PacketProcessor::serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet) const {
  if (packet.globalId() == selfState_.globalId()) {
    std::optional<sro::Position> sourcePosition;
    if (packet.hasSource()) {
      // Server is telling us our source position
      sourcePosition = packet.sourcePosition();
    }
    if (packet.hasDestination()) {
      LOG() << "We are moving to " << packet.destinationPosition() << std::endl;
      selfState_.setMovingToDestination(sourcePosition, packet.destinationPosition());
    } else {
      LOG() << "We are moving toward angle " << packet.angle() << std::endl;
      selfState_.setMovingTowardAngle(sourcePosition, packet.angle());
    }
  } else {
    // Someone other than us is moving
    if (!entityTracker_.trackingEntity(packet.globalId())) {
      throw std::runtime_error("Received movement update for something we're not tracking");
    }
    auto &mobileEntity = entityTracker_.getEntity<entity::MobileEntity>(packet.globalId());
    std::optional<sro::Position> sourcePosition;
    if (packet.hasSource()) {
      // Server is telling us our source position
      sourcePosition = packet.sourcePosition();
    }
    if (packet.hasDestination()) {
      mobileEntity.setMovingToDestination(sourcePosition, packet.destinationPosition(), eventBroker_);
    } else {
      mobileEntity.setMovingTowardAngle(sourcePosition, packet.angle(), eventBroker_);
    }
  }
  return true;
}

// ============================================================================================================================
// ===============================================Character info packet handling===============================================
// ============================================================================================================================

bool PacketProcessor::clientItemMoveReceived(const packet::parsing::ParsedClientItemMove &packet) const {
  const auto itemMovement = packet.movement();
  if (itemMovement.type == packet::enums::ItemMovementType::kBuyItem) {
    // User is buying something from the store
    selfState_.setUserPurchaseRequest(itemMovement);
  }
  return true;
}

bool PacketProcessor::serverAgentCharacterDataReceived(const packet::parsing::ParsedServerAgentCharacterData &packet) const {
  resetDataBecauseCharacterSpawned();

  selfState_.initialize(packet.entityUniqueId(), packet.refObjId());
  selfState_.setHp(packet.hp());
  selfState_.setMp(packet.mp());
  selfState_.setCurrentLevel(packet.curLevel());
  selfState_.setSkillPoints(packet.skillPoints());
  selfState_.setCurrentExpAndSpExp(packet.currentExperience(), packet.currentSpExperience());
  selfState_.setMasteriesAndSkills(packet.masteries(), packet.skills());

  // Position
  selfState_.setStationaryAtPosition(packet.position());
  LOG() << "Our Ref Obj Id " << packet.refObjId() << '\n';
  LOG() << "Position: " << (packet.position().isDungeon() ? "dungeon " : "world ");
  if (packet.position().isDungeon()) {
    std::cout << '#' << (int)packet.position().dungeonId();
  } else {
    std::cout << "region (" << (int)packet.position().xSector() << ',' << (int)packet.position().zSector() << ")";
  }
  std::cout << " (" << packet.position().xOffset() << ',' << packet.position().yOffset() << ',' << packet.position().zOffset() << ")\n";

  // State
  selfState_.setLifeState(packet.lifeState());
  selfState_.setMotionState(packet.motionState());
  selfState_.setBodyState(packet.bodyState());

  // Speed
  selfState_.setSpeed(packet.walkSpeed(), packet.runSpeed());
  selfState_.setHwanSpeed(packet.hwanSpeed());
  selfState_.characterName = packet.characterName();
  auto refObjId = packet.refObjId();
  selfState_.setGold(packet.gold());
  selfState_.setRaceAndGender(refObjId);
  const auto inventorySize = packet.inventorySize();
  const auto &inventoryItemMap = packet.inventoryItemMap();
  helpers::initializeInventory(selfState_.inventory, inventorySize, inventoryItemMap);
  const auto avatarInventorySize = packet.avatarInventorySize();
  const auto &avatarInventoryItemMap = packet.avatarInventoryItemMap();
  helpers::initializeInventory(selfState_.avatarInventory, avatarInventorySize, avatarInventoryItemMap);

  LOG() << "GID:" << selfState_.globalId() << ", and we have " << selfState_.hp() << " hp and " << selfState_.mp() << " mp\n";
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kSpawned));
  return true;
}

bool PacketProcessor::serverAgentCosDataReceived(const packet::parsing::ServerAgentCosData &packet) const {
  if (packet.isAbilityPet()) {
    if (packet.ownerGlobalId() == selfState_.globalId()) {
      auto it = selfState_.cosInventoryMap.find(packet.globalId());
      if (it == selfState_.cosInventoryMap.end()) {
        // Not yet tracking this Cos
        auto emplaceResult = selfState_.cosInventoryMap.emplace(packet.globalId(), storage::Storage());
        if (!emplaceResult.second) {
          throw std::runtime_error("Unable to create new Cos inventory");
        }
        auto &cosInventory = emplaceResult.first->second;
        helpers::initializeInventory(cosInventory, packet.inventorySize(), packet.inventoryItemMap());
        eventBroker_.publishEvent(std::make_unique<event::CosSpawned>(packet.globalId()));
      } else {
        throw std::runtime_error("Aready tracking this Cos");
        // Maybe we should ensure that we never get here
        // On teleport, our COS globalId will change
        // On resummon, our COS globalId will change
      }
    } else {
      LOG() << "Got Cos data for someone else's Cos" << std::endl;
    }
  } else {
    LOG() << "Non-ability Cos" << std::endl;
  }
  return true;
}

bool PacketProcessor::serverAgentInventoryStorageDataReceived(const packet::parsing::ParsedServerAgentInventoryStorageData &packet) const {
  selfState_.setStorageGold(packet.gold());
  helpers::initializeInventory(selfState_.storage, packet.storageSize(), packet.storageItemMap());
  selfState_.haveOpenedStorageSinceTeleport = true;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStorageInitialized));
  return true;
}

bool PacketProcessor::serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet) const {
  if (packet.globalId() == selfState_.globalId()) {
    if (selfState_.spawned()) {
      if (packet.stateType() == packet::parsing::StateType::kBodyState) {
        selfState_.setBodyState(static_cast<packet::enums::BodyState>(packet.state()));
      } else if (packet.stateType() == packet::parsing::StateType::kLifeState) {
        selfState_.setLifeState(static_cast<entity::LifeState>(packet.state()));
        if (static_cast<entity::LifeState>(packet.state()) == entity::LifeState::kDead) {
          LOG() << "CharacterInfoModule: We died, clearing used item queue" << std::endl;
          selfState_.clearUsedItemQueue();
        }
      } else if (packet.stateType() == packet::parsing::StateType::kMotionState) {
        const auto entityMotionState = static_cast<entity::MotionState>(packet.state());
        if (entityMotionState == entity::MotionState::kWalk) {
          LOG() << "Motion state update to walk\n";
        } else if (entityMotionState == entity::MotionState::kRun) {
          LOG() << "Motion state update to run\n";
        } else if (entityMotionState == entity::MotionState::kSit) {
          LOG() << "Motion state update to sit\n";
        } else {
          LOG() << "Motion state update to " << static_cast<int>(packet.state()) << '\n';
        }
        selfState_.setMotionState(entityMotionState);
      }
    } else {
      LOG() << "Got state for ourself, but we're not spawned" << std::endl;
    }
  } else {
    // State for other entity
    if (packet.stateType() == packet::parsing::StateType::kMotionState) {
      auto &mobileEntity = entityTracker_.getEntity<entity::MobileEntity>(packet.globalId());
      mobileEntity.setMotionState(static_cast<entity::MotionState>(packet.state()), eventBroker_);
    } else if (packet.stateType() == packet::parsing::StateType::kLifeState) {
      const auto newLifeState = static_cast<entity::LifeState>(packet.state());
      auto &characterEntity = entityTracker_.getEntity<entity::Character>(packet.globalId());
      if (characterEntity.lifeState != newLifeState) {
        characterEntity.setLifeState(newLifeState, eventBroker_);
      } else {
        LOG() << "Got life state update for entity, but nothing changed" << std::endl;
      }
    }
  }
  return true;
}

bool PacketProcessor::serverAgentEntityUpdateMoveSpeedReceived(const packet::parsing::ServerAgentEntityUpdateMoveSpeed &packet) const {
  if (packet.globalId() == selfState_.globalId()) {
    // Our speed was updated
    LOG() << "Our speed was updated from " << selfState_.walkSpeed() << ',' << selfState_.runSpeed() << " to " << packet.walkSpeed() << ',' << packet.runSpeed() << '\n';
    selfState_.setSpeed(packet.walkSpeed(), packet.runSpeed());
  } else {
    auto &mobileEntity = entityTracker_.getEntity<entity::MobileEntity>(packet.globalId());
    mobileEntity.setSpeed(packet.walkSpeed(), packet.runSpeed(), eventBroker_);
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
    } else {
      LOG() << "Weird, says HP changed, but it didn't\n";
    }
  }
  if (packet.vitalBitmask() & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoMp)) {
    // Our MP changed
    if (selfState_.mp() != packet.newMpValue()) {
      selfState_.setMp(packet.newMpValue());
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
          eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(packet.slotNum() ,std::nullopt));
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
  auto addItemToInventory = [this](auto &inventory, const auto newItem, const auto destSlot) {
    if (newItem != nullptr) {
      // Picked up an item
      if (inventory.hasItem(destSlot)) {
        // There is already something in this slot
        auto existingItem = inventory.getItem(destSlot);
        bool addedToStack = false;
        if (existingItem->refItemId == newItem->refItemId) {
          // Both items have the same refId
          storage::ItemExpendable *newExpendableItem, *existingExpendableItem;
          if ((newExpendableItem = dynamic_cast<storage::ItemExpendable*>(newItem.get())) &&
              (existingExpendableItem = dynamic_cast<storage::ItemExpendable*>(existingItem))) {
            // Both items are expendables, so we can stack them
            // Picked item's quantity (if an expendable) is the total in the given slot
            existingExpendableItem->quantity = newExpendableItem->quantity;
            addedToStack = true;
            eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, destSlot));
          }
        }
        if (!addedToStack) {
          LOG() << "Error: Item couldnt be added to the stack\n";
        }
      } else {
        // This is a new item
        inventory.addItem(destSlot, newItem);
        if (!inventory.hasItem(destSlot)) {
          // This is especially weird since we already know that there was nothing in this slot
          throw std::runtime_error("Could not add item to inventory");
        }
        eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, destSlot));
      }
    } else {
      LOG() << "Error: Picked an item, but the newItem is a nullptr\n";
    }
  };

  auto removeItemFromInventory = [this](const auto slotIndex) {
    if (selfState_.inventory.hasItem(slotIndex)) {
      auto itemPtr = selfState_.inventory.withdrawItem(slotIndex);
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(slotIndex, std::nullopt));
    } else {
      LOG() << "Error: But there's no item in this inventory slot\n";
    }
  };

  // TODO: If we used an item and it moved, we'll need to update the "reference" to this item in the used item queue
  const std::vector<packet::structures::ItemMovement> &itemMovements = packet.itemMovements();
  for (const auto &movement : itemMovements) {
    if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsInventory) {
      selfState_.inventory.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsChest) {
      selfState_.storage.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent(std::make_unique<event::StorageUpdated>(movement.srcSlot, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsGuildChest) {
      selfState_.guildStorage.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent(std::make_unique<event::GuildStorageUpdated>(movement.srcSlot, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kChestDepositItem) {
      selfState_.storage.addItem(movement.destSlot, selfState_.inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::StorageUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kChestWithdrawItem) {
      selfState_.inventory.addItem(movement.destSlot, selfState_.storage.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::StorageUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestDepositItem) {
      selfState_.guildStorage.addItem(movement.destSlot, selfState_.inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::GuildStorageUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestWithdrawItem) {
      selfState_.inventory.addItem(movement.destSlot, selfState_.guildStorage.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::GuildStorageUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kBuyItem) {
      if (selfState_.haveUserPurchaseRequest()) {
        const auto userPurchaseRequest = selfState_.getUserPurchaseRequest();
        // User purchased something, we saved this so that we can get the NPC's global Id
        if (entityTracker_.trackingEntity(userPurchaseRequest.globalId)) {
          auto object = entityTracker_.getEntity(userPurchaseRequest.globalId);
          // Found the NPC which this purchase was made with
          if (gameData_.characterData().haveCharacterWithId(object->refObjId)) {
            auto npcName = gameData_.characterData().getCharacterById(object->refObjId).codeName128;
            auto itemInfo = gameData_.shopData().getItemFromNpc(npcName, userPurchaseRequest.storeTabNumber, userPurchaseRequest.storeSlotNumber);
            const auto &itemRef = gameData_.itemData().getItemByCodeName128(itemInfo.refItemCodeName);
            if (movement.destSlots.size() == 1) {
              // Just a single item or single stack
              auto item = helpers::createItemFromScrap(itemInfo, itemRef);
              storage::ItemExpendable *itemExp = dynamic_cast<storage::ItemExpendable*>(item.get());
              if (itemExp != nullptr) {
                itemExp->quantity = movement.quantity;
              }
              selfState_.inventory.addItem(movement.destSlots[0], item);
              eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlots[0]));
            } else {
              // Multiple destination slots, must be unstackable items like equipment
              for (auto destSlot : movement.destSlots) {
                auto item = helpers::createItemFromScrap(itemInfo, itemRef);
                selfState_.inventory.addItem(destSlot, item);
                eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
              }
            }
          }
        }
        selfState_.resetUserPurchaseRequest();
      } else {
        LOG() << "kBuyItem but we dont have the data from the client packet\n";
        // TODO: Introduce unknown item concept?
      }
    } else if (movement.type == packet::enums::ItemMovementType::kSellItem) {
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
        eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, std::nullopt));
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
                eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
              }
            }
          }
          if (boughtBackAll) {
            selfState_.inventory.addItem(movement.destSlot, selfState_.buybackQueue.withdrawItem(movement.srcSlot));
            eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
          }
        } else {
          LOG() << "Bought back item is being moved into a slot that's already occupied\n";
        }
      } else {
        LOG() << "Bought back an item that we werent tracking\n";
      }
    } else if (movement.type == packet::enums::ItemMovementType::kPickItem) {
      if (movement.destSlot != packet::structures::ItemMovement::kGoldSlot) {
        addItemToInventory(selfState_.inventory, movement.newItem, movement.destSlot);
      }
      // This would be a good time to try to use a pill, potion, return scroll, etc.
    } else if (movement.type == packet::enums::ItemMovementType::kDropItem) {
      removeItemFromInventory(movement.srcSlot);
    } else if (movement.type == packet::enums::ItemMovementType::kAddItemByServer) {
      addItemToInventory(selfState_.inventory, movement.newItem, movement.destSlot);
    } else if (movement.type == packet::enums::ItemMovementType::kRemoveItemByServer) {
      removeItemFromInventory(movement.srcSlot);
    } else if (movement.type == packet::enums::ItemMovementType::kDropGold) {
      // Another packet, ServerAgentEntityUpdatePoints, contains character gold update information
    } else if (movement.type == packet::enums::ItemMovementType::kChestWithdrawGold) {
      selfState_.setStorageGold(selfState_.getStorageGold() - movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kChestDepositGold) {
      selfState_.setStorageGold(selfState_.getStorageGold() + movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestDepositGold) {
      selfState_.setGuildStorageGold(selfState_.getGuildStorageGold() - movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestWithdrawGold) {
      selfState_.setGuildStorageGold(selfState_.getGuildStorageGold() + movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemAvatarToInventory) {
      selfState_.inventory.addItem(movement.destSlot, selfState_.avatarInventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::AvatarInventoryUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemInventoryToAvatar) {
      selfState_.avatarInventory.addItem(movement.destSlot, selfState_.inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::AvatarInventoryUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemCosToInventory) {
      auto &cosInventory = selfState_.getCosInventory(movement.globalId);
      selfState_.inventory.addItem(movement.destSlot, cosInventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::CosInventoryUpdated>(movement.globalId, movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemInventoryToCos) {
      auto &cosInventory = selfState_.getCosInventory(movement.globalId);
      cosInventory.addItem(movement.destSlot, selfState_.inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::CosInventoryUpdated>(movement.globalId, std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsInventoryCos) {
      auto &cosInventory = selfState_.getCosInventory(movement.globalId);
      cosInventory.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent(std::make_unique<event::CosInventoryUpdated>(movement.globalId, movement.srcSlot, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kPickItemCos) {
      auto &cosInventory = selfState_.getCosInventory(movement.globalId);
      addItemToInventory(cosInventory, movement.newItem, movement.destSlot);
      eventBroker_.publishEvent(std::make_unique<event::CosInventoryUpdated>(movement.globalId, std::nullopt, movement.destSlot));
    } else {
      LOG() << "Unknown item movement type: " << static_cast<int>(movement.type) << std::endl;
    }
  }
  return true;
}

bool PacketProcessor::serverAgentEntityGroupSpawnDataReceived(const packet::parsing::ParsedServerAgentEntityGroupSpawnData &packet) const {
  if (packet.groupSpawnType() == packet::parsing::GroupSpawnType::kSpawn) {
    for (auto entity : packet.entities()) {
      entitySpawned(entity);
    }
  } else {
    for (auto globalId : packet.despawnGlobalIds()) {
      entityDespawned(globalId);
    }
  }
  return true;
}

bool PacketProcessor::serverAgentSpawnReceived(const packet::parsing::ParsedServerAgentSpawn &packet) const {
  if (packet.entity()) {
    entitySpawned(packet.entity());
  } else {
    LOG() << "Object spawned which we cannot track\n";
  }
  return true;
}

bool PacketProcessor::serverAgentDespawnReceived(const packet::parsing::ParsedServerAgentDespawn &packet) const {
  entityDespawned(packet.globalId());
  return true;
}

void PacketProcessor::entitySpawned(std::shared_ptr<entity::Entity> entity) const {
  entityTracker_.trackEntity(entity);
  eventBroker_.publishEvent(std::make_unique<event::EntitySpawned>(entity->globalId));

  // Check if the entity spawned in as already moving
  auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(entity.get());
  if (mobileEntity == nullptr) {
    // Non-mobile, nothing to do
    return;
  }
  if (mobileEntity->moving) {
    if (mobileEntity->destinationPosition) {
      // Entity spawned and is moving to a destination
      mobileEntity->setMovingToDestination(mobileEntity->position(), *mobileEntity->destinationPosition, eventBroker_);
    } else if (mobileEntity->movementAngle) {
      mobileEntity->setMovingTowardAngle(mobileEntity->position(), *mobileEntity->movementAngle, eventBroker_);
    } else {
      throw std::runtime_error("Entity is moving, but has no destination position nor angle");
    }
  }
}

void PacketProcessor::entityDespawned(sro::scalar_types::EntityGlobalId globalId) const {
  // Before destroying an entity, see if we have a running movement timer to cancel
  auto *entity = entityTracker_.getEntity(globalId);
  auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(entity);
  if (mobileEntity != nullptr) {
    // Is a mobile entity
    if (mobileEntity->movingEventId) {
      eventBroker_.cancelDelayedEvent(*mobileEntity->movingEventId);
    }
  }

  // Destroy entity
  entityTracker_.stopTrackingEntity(globalId);
  eventBroker_.publishEvent(std::make_unique<event::EntityDespawned>(globalId));
}

// ============================================================================================================================
// ============================================================Misc============================================================
// ============================================================================================================================

bool PacketProcessor::serverAgentDeselectResponseReceived(const packet::parsing::ServerAgentActionDeselectResponse &packet) const {
  if (packet.result() == 1) {
    // Successfully deselected
    // If there is a talk dialog, and we have an npc selected, it will take 2 deselects to close both dialogs
    //  First, the talk dialog is closed
    if (selfState_.talkingGidAndOption) {
      // This closes the talk dialog
      selfState_.talkingGidAndOption.reset();
      eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kEntityDeselected));
    } else {
      //  The entity is deselected
      if (selfState_.selectedEntity) {
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
    // It is possible that we already have something selected. We will just overwrite it
    selfState_.selectedEntity = packet.gId();
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kEntitySelected));
  } else {
    LOG() << "Selection failed" << std::endl;
  }
  return true;
}

bool PacketProcessor::serverAgentTalkResponseReceived(const packet::parsing::ServerAgentActionTalkResponse &packet) const {
  if (packet.result() == 1) {
    // Successfully talking to an npc
    if (selfState_.pendingTalkGid) {
      // We were waiting for this response
      selfState_.talkingGidAndOption = std::make_pair(*selfState_.pendingTalkGid, packet.talkOption());
      selfState_.pendingTalkGid.reset();
      eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kNpcTalkStart));
    } else {
      LOG() << "Weird, we werent expecting to be talking to anything. As a result, we dont know what we're talking to" << std::endl;
    }
  } else {
    LOG() << "Failed to talk to NPC" << std::endl;
  }
  return true;
}

bool PacketProcessor::serverAgentInventoryRepairResponseReceived(const packet::parsing::ServerAgentInventoryRepairResponse &packet) const {
  if (packet.successful()) {
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kRepairSuccessful));
  } else {
    LOG() << "Repairing item(s) failed! Error code: " << packet.errorCode() << std::endl;
  }
  return true;
}

bool PacketProcessor::serverAgentInventoryUpdateDurabilityReceived(const packet::parsing::ServerAgentInventoryUpdateDurability &packet) const {
  if (!selfState_.inventory.hasItem(packet.slotIndex())) {
    throw std::runtime_error("Recieved durability update for inventory slot where no item exists");
  }
  auto *item = selfState_.inventory.getItem(packet.slotIndex());
  if (item == nullptr) {
    throw std::runtime_error("Recieved durability update for inventory item which is null");
  }
  auto *itemAsEquip = dynamic_cast<storage::ItemEquipment*>(item);
  if (itemAsEquip == nullptr) {
    throw std::runtime_error("Recieved durability update for inventory item which is not a piece of equipment");
  }
  // Update item's durability
  itemAsEquip->durability = packet.durability();
  return true;
}

bool PacketProcessor::serverAgentInventoryUpdateItemReceived(const packet::parsing::ServerAgentInventoryUpdateItem &packet) const {
  if (!selfState_.inventory.hasItem(packet.slotIndex())) {
    throw std::runtime_error("Recieved item update for inventory slot where no item exists");
  }
  auto *item = selfState_.inventory.getItem(packet.slotIndex());
  if (item == nullptr) {
    throw std::runtime_error("Recieved item update for inventory item which is null");
  }
  if (packet.itemUpdateHasFlag(packet::enums::ItemUpdateFlag::kQuantity)) {
    // Known reasons for this update: alchemy
    // Try to cast item as expendable
    if (auto *itemAsExpendable = dynamic_cast<storage::ItemExpendable*>(item)) {
      const bool increased = (packet.quantity() > itemAsExpendable->quantity);
      itemAsExpendable->quantity = packet.quantity();
      if (increased) {
        eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, packet.slotIndex()));
      } else {
        eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(packet.slotIndex() ,std::nullopt));
      }
    } else {
      throw std::runtime_error("Item quantity updated, but this item is not an expendable");
    }
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
  if (selfState_.pendingTalkGid) {
    LOG() << "Weird, we're already waiting on a response from the server to talk to someone\n";
  } else {
    selfState_.pendingTalkGid = packet.gId();
  }
  return true;
}

bool PacketProcessor::serverAgentEntityUpdatePointsReceived(const packet::parsing::ServerAgentEntityUpdatePoints &packet) const {
  if (packet.updatePointsType() == packet::enums::UpdatePointsType::kGold) {
    selfState_.setGold(packet.gold());
  } else if (packet.updatePointsType() == packet::enums::UpdatePointsType::kSp) {
    selfState_.setSkillPoints(packet.skillPoints());
  }
  return true;
}

bool PacketProcessor::serverAgentEntityUpdateExperienceReceived(const packet::parsing::ServerAgentEntityUpdateExperience &packet) const {
  const constexpr int kSpExperienceRequired{400}; // TODO: Move to a central location
  const auto newExperience = selfState_.getCurrentExperience() + packet.gainedExperiencePoints();
  const auto newSpExperience = (selfState_.getCurrentSpExperience() + packet.gainedSpExperiencePoints()) % kSpExperienceRequired;
  selfState_.setCurrentExpAndSpExp(newExperience, newSpExperience);
  return true;
}

bool PacketProcessor::serverAgentGuildStorageDataReceived(const packet::parsing::ServerAgentGuildStorageData &packet) const {
  selfState_.setGuildStorageGold(packet.gold());
  helpers::initializeInventory(selfState_.guildStorage, packet.storageSize(), packet.storageItemMap());
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kGuildStorageInitialized));
  return true;
}

bool PacketProcessor::serverAgentActionCommandResponseReceived(const packet::parsing::ServerAgentActionCommandResponse &packet) const {
  LOG() << "serverAgentActionCommandResponseReceived" << std::endl;
  return true;
}

bool PacketProcessor::serverAgentSkillBeginReceived(const packet::parsing::ServerAgentSkillBegin &packet) const {
  LOG() << "serverAgentSkillBeginReceived" << std::endl;
  // TODO: Check if this skill can be cast while moving or not
  if (packet.casterGlobalId() == selfState_.globalId()) {
    // We casted this skill
    LOG() << "We're the one who cast the skill" << std::endl;
    if (!gameData_.skillData().haveSkillWithId(packet.refSkillId())) {
      throw std::runtime_error("Cast a skill which we dont have data for");
    }
    const auto &skill = gameData_.skillData().getSkillById(packet.refSkillId());
    switch (skill.basicActivity) {
      case 0:
        // Seems to be passives
        LOG() << "Cast a skill with basic activity == 0" << std::endl;
        break;
      case 1:
        // Dont stop while running. Can be cast while something else is being case
        LOG() << "Cast a skill which does not affect movement" << std::endl;
        break;
      case 2:
        // Will stop you if you're running
        LOG() << "Cast a skill which stops us";
        if (selfState_.moving()) {
          selfState_.setStationaryAtPosition(selfState_.position());
        } else {
          std::cout << " but we werent moving";
        }
        std::cout << std::endl;
        break;
      default:
        LOG() << "Cast a skill with unknown basic activity == " << skill.basicActivity << std::endl;
        break;
    }
  }
  return true;
}

bool PacketProcessor::serverAgentSkillEndReceived(const packet::parsing::ServerAgentSkillEnd &packet) const {
  LOG() << "serverAgentSkillEndReceived" << std::endl;
  return true;
}