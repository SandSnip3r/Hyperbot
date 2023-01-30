#include "packetProcessor.hpp"

#include "entity/entity.hpp"
#include "helpers.hpp"
#include "logging.hpp"
// TODO: <remove>
// For quicker development, when we spawn in, set ourself as visible and put on a PVP cape
#include "packet/building/clientAgentFreePvpUpdateRequest.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"
// </remove>
#include "packet/building/clientAgentActionCommandRequest.hpp"
#include "packet/opcode.hpp"

#include <silkroad_lib/position.h>

#define TRY_CAST_AND_HANDLE_PACKET(PACKET_TYPE, HANDLE_FUNCTION_NAME) \
{ \
  auto *castedParsedPacket = dynamic_cast<PACKET_TYPE*>(parsedPacket.get()); \
  if (castedParsedPacket != nullptr) { \
    HANDLE_FUNCTION_NAME(*castedParsedPacket); \
    return; \
  } \
}

PacketProcessor::PacketProcessor(state::WorldState &worldState,
                                 broker::PacketBroker &brokerSystem,
                                 broker::EventBroker &eventBroker,
                                 const pk2::GameData &gameData) :
      worldState_(worldState),
      packetBroker_(brokerSystem),
      eventBroker_(eventBroker),
      gameData_(gameData) {
  subscribeToPackets();
}

void PacketProcessor::subscribeToPackets() {
  auto packetHandleFunction = std::bind(&PacketProcessor::handlePacket, this, std::placeholders::_1);

  // Server packets
  //   Login packets
  packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentAuthRequest, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerGatewayShardListResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerGatewayLoginResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::LOGIN_CLIENT_INFO, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentAuthResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterSelectionActionResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterSelectionJoinResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerGatewayLoginIbuvChallenge, packetHandleFunction);
  // packetBroker_.subscribeToServerPacket(static_cast<packet::Opcode>(0x6005), packetHandleFunction);
  //   Movement packets
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateAngle, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateMovement, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdatePosition, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntitySyncPosition, packetHandleFunction);
  //   Character info packets
  packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentInventoryOperationRequest, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterData, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCosData, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryStorageData, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateState, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateStatus, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentAbnormalInfo, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterUpdateStats, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryItemUseResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryOperationResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateMoveSpeed, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityRemoveOwnership, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityGroupspawnData, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntitySpawn, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityDespawn, packetHandleFunction);

  //   Misc. packets
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionDeselectResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionSelectResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionTalkResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryRepairResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryUpdateDurability, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentInventoryUpdateItem, packetHandleFunction);
  // packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionDeselectRequest, packetHandleFunction);
  // packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionSelectRequest, packetHandleFunction);
  packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionTalkRequest, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdatePoints, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateExperience, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentGuildStorageData, packetHandleFunction);
  packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionCommandRequest, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionCommandResponse, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillBegin, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillEnd, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentBuffAdd, packetHandleFunction);
  packetBroker_.subscribeToServerPacket(packet::Opcode::kServerAgentBuffRemove, packetHandleFunction);
  packetBroker_.subscribeToClientPacket(packet::Opcode::kClientAgentInventoryItemUseRequest, packetHandleFunction);
}

void PacketProcessor::handlePacket(const PacketContainer &packet) const {
  std::unique_ptr<packet::parsing::ParsedPacket> parsedPacket;
  try {
    parsedPacket = packetParser_.parsePacket(packet);
  } catch (std::exception &ex) {
    LOG() << "Failed to parse packet " << std::hex << packet.opcode << std::dec << ". \"" << ex.what() << '"' << std::endl;
    return;
  }

  if (!parsedPacket) {
    // Not yet parsing this packet
    LOG() << "Subscribed to a packet which we're not yet parsing " << std::hex << packet.opcode << std::dec << std::endl;
    return;
  }

  std::unique_lock<std::mutex> selfStateLock(worldState_.selfState().selfMutex);

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
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentCharacterData, serverAgentCharacterDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentCosData, serverAgentCosDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentInventoryStorageData, serverAgentInventoryStorageDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateState, serverAgentEntityUpdateStateReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateMoveSpeed, serverAgentEntityUpdateMoveSpeedReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityRemoveOwnership, serverAgentEntityRemoveOwnershipReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityUpdateStatus, serverAgentEntityUpdateStatusReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentAbnormalInfo, serverAgentAbnormalInfoReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ParsedServerAgentCharacterUpdateStats, serverAgentCharacterUpdateStatsReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryItemUseResponse, serverAgentInventoryItemUseResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentInventoryOperationResponse, serverAgentInventoryOperationResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityGroupSpawnData, serverAgentEntityGroupSpawnDataReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntitySpawn, serverAgentEntitySpawnReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentEntityDespawn, serverAgentEntityDespawnReceived);

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

    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentActionCommandRequest, clientAgentActionCommandRequestReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentActionCommandResponse, serverAgentActionCommandResponseReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentSkillBegin, serverAgentSkillBeginReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentSkillEnd, serverAgentSkillEndReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentBuffAdd, serverAgentBuffAddReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ServerAgentBuffRemove, serverAgentBuffRemoveReceived);
    TRY_CAST_AND_HANDLE_PACKET(packet::parsing::ClientAgentInventoryItemUseRequest, clientAgentInventoryItemUseRequestReceived);
  } catch (std::exception &ex) {
    LOG() << "Error while handling packet!\n  " << ex.what() << std::endl;
  }

  LOG() << "Unhandled packet subscribed to " << std::hex << packet.opcode << std::dec << std::endl;
  return;
}

void PacketProcessor::resetDataBecauseCharacterSpawned() const {
  // On teleport, COS will have different globaIds
  worldState_.selfState().cosInventoryMap.clear();
}

// ============================================================================================================================
// ===============================================Login process packet handling================================================
// ============================================================================================================================

void PacketProcessor::serverListReceived(const packet::parsing::ParsedLoginServerList &packet) const {
  worldState_.selfState().shardId = packet.shardId();
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateShardIdUpdated));
}

void PacketProcessor::loginResponseReceived(const packet::parsing::ParsedLoginResponse &packet) const {
  if (packet.result() == packet::enums::LoginResult::kSuccess) {
    worldState_.selfState().token = packet.token();
  } else {
    LOG() << " Login failed\n";
  }
}

void PacketProcessor::loginClientInfoReceived(const packet::parsing::ParsedLoginClientInfo &packet) const {
  // This packet is a response to the client sending 0x2001 where the client indicates that it is the "SR_Client"
  if (packet.serviceName() == "AgentServer") {
    // Connected to agentserver, send client auth packet
    worldState_.selfState().connectedToAgentServer = true;
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateConnectedToAgentServerUpdated));
  }
}

void PacketProcessor::unknownPacketReceived(const packet::parsing::ParsedUnknown &packet) const {
  if (packet.opcode() == packet::Opcode::kServerGatewayLoginIbuvChallenge) {
    // Got the captcha prompt, respond with an answer
    worldState_.selfState().receivedCaptchaPrompt = true;
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateReceivedCaptchaPromptUpdated));
  }
}

void PacketProcessor::serverAuthReceived(const packet::parsing::ParsedServerAuthResponse &packet) const {
  if (packet.result() == 0x01) {
    // Successful login
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kLoggedIn));
    // Client will automatically request the character listing
    // TODO: For clientless, we will need to do this ourself
  }
}

void PacketProcessor::charListReceived(const packet::parsing::ParsedServerAgentCharacterSelectionActionResponse &packet) const {
  worldState_.selfState().characterList = packet.characters();
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateCharacterListUpdated));
}

void PacketProcessor::charSelectionJoinResponseReceived(const packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse &packet) const {
  // A character was selected after login, this is the response
  if (packet.result() != 0x01) {
    // Character selection failed
    // TODO: Properly handle error
    LOG() << "Failed when selecting character\n";
  }
}

// ============================================================================================================================
// ==================================================Movement packet handling==================================================
// ============================================================================================================================

void PacketProcessor::serverAgentEntityUpdateAngleReceived(packet::parsing::ServerAgentEntityUpdateAngle &packet) const {
  entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
  if (mobileEntity.moving()) {
    if (mobileEntity.destinationPosition) {
      throw std::runtime_error("Got angle update, but we're running to a destination");
    }
    if (mobileEntity.angle() != packet.angle()) {
      // Changed angle while running
      mobileEntity.setMovingTowardAngle(std::nullopt, packet.angle(), eventBroker_);
    }
  } else {
    mobileEntity.setAngle(packet.angle(), eventBroker_);
  }
}

void PacketProcessor::serverAgentEntitySyncPositionReceived(packet::parsing::ServerAgentEntitySyncPosition &packet) const {
  entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
  mobileEntity.syncPosition(packet.position(), eventBroker_);
}

void PacketProcessor::serverAgentEntityUpdatePositionReceived(packet::parsing::ServerAgentEntityUpdatePosition &packet) const {
  entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
  mobileEntity.setStationaryAtPosition(packet.position(), eventBroker_);
}

void PacketProcessor::serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet) const {
  entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
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

// ============================================================================================================================
// ===============================================Character info packet handling===============================================
// ============================================================================================================================

void PacketProcessor::clientItemMoveReceived(const packet::parsing::ParsedClientItemMove &packet) const {
  const auto itemMovement = packet.movement();
  if (itemMovement.type == packet::enums::ItemMovementType::kBuyItem) {
    // User is buying something from the store
    worldState_.selfState().setUserPurchaseRequest(itemMovement);
  }
}

void PacketProcessor::serverAgentCharacterDataReceived(const packet::parsing::ServerAgentCharacterData &packet) const {
  resetDataBecauseCharacterSpawned();

  worldState_.selfState().initialize(packet.entityUniqueId(), packet.refObjId(), packet.jId());
  worldState_.selfState().initializeCurrentHp(packet.hp());
  worldState_.selfState().setMp(packet.mp());
  worldState_.selfState().setCurrentLevel(packet.curLevel());
  worldState_.selfState().setSkillPoints(packet.skillPoints());
  worldState_.selfState().setCurrentExpAndSpExp(packet.currentExperience(), packet.currentSpExperience());
  worldState_.selfState().setMasteriesAndSkills(packet.masteries(), packet.skills());

  // Position
  // TODO: Handle the case when the character spawns in a moving state
  worldState_.selfState().setStationaryAtPosition(packet.position(), eventBroker_);
  worldState_.selfState().initializeAngle(packet.angle());

  // State
  worldState_.selfState().setLifeState(packet.lifeState(), eventBroker_);
  worldState_.selfState().setMotionState(packet.motionState(), eventBroker_);
  worldState_.selfState().setBodyState(packet.bodyState());

  // Buffs
  // TODO: If we spawn with any active buffs, add them
  // worldState_.addBuff(packet.globalId(), packet.skillRefId(), packet.activeBuffToken());

  // Speed
  worldState_.selfState().setSpeed(packet.walkSpeed(), packet.runSpeed(), eventBroker_);
  worldState_.selfState().setHwanSpeed(packet.hwanSpeed());
  worldState_.selfState().name = packet.characterName();
  worldState_.selfState().setGold(packet.gold());
  const auto inventorySize = packet.inventorySize();
  const auto &inventoryItemMap = packet.inventoryItemMap();
  helpers::initializeInventory(worldState_.selfState().inventory, inventorySize, inventoryItemMap);
  const auto avatarInventorySize = packet.avatarInventorySize();
  const auto &avatarInventoryItemMap = packet.avatarInventoryItemMap();
  helpers::initializeInventory(worldState_.selfState().avatarInventory, avatarInventorySize, avatarInventoryItemMap);

  LOG() << "GID:" << worldState_.selfState().globalId << ", and we have " << worldState_.selfState().currentHp() << " hp and " << worldState_.selfState().currentMp() << " mp\n";
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kSpawned));
}

void PacketProcessor::serverAgentCosDataReceived(const packet::parsing::ServerAgentCosData &packet) const {
  if (packet.isAbilityPet()) {
    if (packet.ownerGlobalId() == worldState_.selfState().globalId) {
      // Is our pickpet
      auto it = worldState_.selfState().cosInventoryMap.find(packet.globalId());
      if (it == worldState_.selfState().cosInventoryMap.end()) {
        // Not yet tracking this Cos
        auto emplaceResult = worldState_.selfState().cosInventoryMap.emplace(packet.globalId(), storage::Storage());
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
}

void PacketProcessor::serverAgentInventoryStorageDataReceived(const packet::parsing::ParsedServerAgentInventoryStorageData &packet) const {
  worldState_.selfState().setStorageGold(packet.gold());
  helpers::initializeInventory(worldState_.selfState().storage, packet.storageSize(), packet.storageItemMap());
  worldState_.selfState().haveOpenedStorageSinceTeleport = true;
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStorageInitialized));
}

void PacketProcessor::serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet) const {
  if (packet.stateType() == packet::parsing::StateType::kMotionState) {
    entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
    mobileEntity.setMotionState(static_cast<entity::MotionState>(packet.state()), eventBroker_);
  } else if (packet.stateType() == packet::parsing::StateType::kLifeState) {
    entity::Character &characterEntity = worldState_.getEntity<entity::Character>(packet.globalId());
    if (packet.globalId() == worldState_.selfState().globalId && !worldState_.selfState().spawned()) {
      throw std::runtime_error("Got life state update for ourself, but we are not spawned");
    }
    const auto newLifeState = static_cast<sro::entity::LifeState>(packet.state());
    characterEntity.setLifeState(newLifeState, eventBroker_);
  } else {
    if (!worldState_.selfState().spawned()) {
      throw std::runtime_error("Got state update for ourself, but we are not spawned");
    }
    if (packet.stateType() == packet::parsing::StateType::kBodyState) {
      worldState_.selfState().setBodyState(static_cast<packet::enums::BodyState>(packet.state()));
    }
  }
  
  // TODO: Remove
  // For quicker development, when we spawn in, set ourself as visible and put on a PVP cape
  if (packet.globalId() == worldState_.selfState().globalId) {
    static bool done = false;
    if (!done) {
      // Just spawned in
      // Set self as visible
      LOG() << "Setting self as visible" << std::endl;
      const auto setVisiblePacket = packet::building::ClientAgentOperatorRequest::toggleInvisible();
      packetBroker_.injectPacket(setVisiblePacket, PacketContainer::Direction::kClientToServer);
      
      // LOG() << "Setting free pvp mode" << std::endl;
      // const auto setPvpModePacket = packet::building::ClientAgentFreePvpUpdateRequest::setMode(packet::enums::FreePvpMode::kYellow);
      // packetBroker_.injectPacket(setPvpModePacket, PacketContainer::Direction::kClientToServer);

      done = true;
    }
  }
}

void PacketProcessor::serverAgentEntityUpdateMoveSpeedReceived(const packet::parsing::ServerAgentEntityUpdateMoveSpeed &packet) const {
  entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.globalId());
  mobileEntity.setSpeed(packet.walkSpeed(), packet.runSpeed(), eventBroker_);
}

void PacketProcessor::serverAgentEntityRemoveOwnershipReceived(const packet::parsing::ServerAgentEntityRemoveOwnership &packet) const {
  entity::Item &itemEntity = worldState_.getEntity<entity::Item>(packet.globalId());
  itemEntity.removeOwnership(eventBroker_);
}

void PacketProcessor::serverAgentEntityUpdateStatusReceived(const packet::parsing::ServerAgentEntityUpdateStatus &packet) const {
  if (packet.entityUniqueId() == worldState_.selfState().globalId) {
    if (flags::isSet(packet.vitalBitmask(), packet::enums::VitalInfoFlag::kVitalInfoHp)) {
      // Our HP changed
      if (worldState_.selfState().currentHp() != packet.newHpValue()) {
        worldState_.selfState().setCurrentHp(packet.newHpValue(), eventBroker_);
      }
    }
    if (flags::isSet(packet.vitalBitmask(), packet::enums::VitalInfoFlag::kVitalInfoMp)) {
      // Our MP changed
      if (worldState_.selfState().currentMp() != packet.newMpValue()) {
        worldState_.selfState().setMp(packet.newMpValue());
      }
    }

    if (flags::isSet(packet.vitalBitmask(), packet::enums::VitalInfoFlag::kVitalInfoAbnormal)) {
      // Our states changed
      auto stateBitmask = packet.stateBitmask();
      auto stateLevels = packet.stateLevels();
      worldState_.selfState().updateStates(stateBitmask, stateLevels);
      eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStatesChanged));
    }
  } else {
    // Not for my character
    if (flags::isSet(packet.vitalBitmask(), packet::enums::VitalInfoFlag::kVitalInfoHp)) {
      auto &character = worldState_.getEntity<entity::Character>(packet.entityUniqueId());
      character.setCurrentHp(packet.newHpValue(), eventBroker_);
    }
  }
}

void PacketProcessor::serverAgentAbnormalInfoReceived(const packet::parsing::ParsedServerAgentAbnormalInfo &packet) const {
  for (int i=0; i<=helpers::toBitNum(packet::enums::AbnormalStateFlag::kZombie); ++i) {
    worldState_.selfState().setLegacyStateEffect(helpers::fromBitNum(i), packet.states()[i].effectOrLevel);
  }
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStatesChanged));
}

void PacketProcessor::serverAgentCharacterUpdateStatsReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet) const {
  worldState_.selfState().setMaxHpMp(packet.maxHp(), packet.maxMp());
}

void PacketProcessor::serverAgentInventoryItemUseResponseReceived(const packet::parsing::ServerAgentInventoryItemUseResponse &packet) const {
  if (packet.result() == 1) {
    // Successfully used an item
    // Make sure we have the item
    if (!worldState_.selfState().inventory.hasItem(packet.slotNum())) {
      throw std::runtime_error("Used an item, but it's not in our inventory");
    }

    auto *itemPtr = worldState_.selfState().inventory.getItem(packet.slotNum());
    // Lets double check its type data
    if (packet.typeData() != itemPtr->typeData()) {
      throw std::runtime_error("Used an item, but the stored typeData doesn't match what came in the packet");
    }

    auto *expendableItemPtr = dynamic_cast<storage::ItemExpendable*>(itemPtr);
    // Make sure that the item is an expendable
    if (expendableItemPtr == nullptr) {
      throw std::runtime_error("Used an item, but it wasn't an expendable");
    }

    expendableItemPtr->quantity = packet.remainingCount();
    worldState_.selfState().usedAnItem(packet.typeData(), eventBroker_);
    eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(packet.slotNum(), std::nullopt));
  } else {
    // Failed to use item
    if (!worldState_.selfState().usedItemQueueIsEmpty()) {
      // This was an item that we tried to use
      if (packet.errorCode() != packet::enums::InventoryErrorCode::kWaitForReuseDelay &&
          packet.errorCode() != packet::enums::InventoryErrorCode::kCharacterDead &&
          packet.errorCode() != packet::enums::InventoryErrorCode::kItemDoesNotExist) {
        LOG() << "Unknown error while trying to use an item: " << static_cast<int>(packet.errorCode()) << '\n';
      }
      const auto usedItem = worldState_.selfState().getUsedItemQueueFront();
      eventBroker_.publishEvent(std::make_unique<event::ItemUseFailed>(usedItem.inventorySlotNum, usedItem.typeId, packet.errorCode()));
    } else {
      // We subscribe to the client item use packet to fill the used item queue. If there isnt an item here, something is weird
      LOG() << "SHOULDNT HAPPEN. Used item queue is empty!" << std::endl;
    }
  }
  if (!worldState_.selfState().usedItemQueueIsEmpty()) {
    // TODO: Refactor this whole itemUsedTimeout concept
    // Received a response for our item
    if (worldState_.selfState().itemUsedTimeoutTimer) {
      eventBroker_.cancelDelayedEvent(*worldState_.selfState().itemUsedTimeoutTimer);
      worldState_.selfState().itemUsedTimeoutTimer.reset();
    }
  } else {
    LOG() << "SHOULDNT HAPPEN. Used item queue be empty!" << std::endl;
  }
  worldState_.selfState().popItemFromUsedItemQueueIfNotEmpty();
}

void PacketProcessor::serverAgentInventoryOperationResponseReceived(const packet::parsing::ServerAgentInventoryOperationResponse &packet) const {
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
    if (worldState_.selfState().inventory.hasItem(slotIndex)) {
      worldState_.selfState().inventory.deleteItem(slotIndex);
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(slotIndex, std::nullopt));
    } else {
      LOG() << "RemoveItemFromInventory(): There's no item in this inventory slot\n";
    }
  };

  // TODO: If we used an item and it moved, we'll need to update the "reference" to this item in the used item queue
  const std::vector<packet::structures::ItemMovement> &itemMovements = packet.itemMovements();
  for (const auto &movement : itemMovements) {
    if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsInventory) {
      worldState_.selfState().inventory.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsChest) {
      worldState_.selfState().storage.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent(std::make_unique<event::StorageUpdated>(movement.srcSlot, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsGuildChest) {
      worldState_.selfState().guildStorage.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent(std::make_unique<event::GuildStorageUpdated>(movement.srcSlot, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kChestDepositItem) {
      worldState_.selfState().storage.addItem(movement.destSlot, worldState_.selfState().inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::StorageUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kChestWithdrawItem) {
      worldState_.selfState().inventory.addItem(movement.destSlot, worldState_.selfState().storage.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::StorageUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestDepositItem) {
      worldState_.selfState().guildStorage.addItem(movement.destSlot, worldState_.selfState().inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::GuildStorageUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestWithdrawItem) {
      worldState_.selfState().inventory.addItem(movement.destSlot, worldState_.selfState().guildStorage.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::GuildStorageUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kBuyItem) {
      if (worldState_.selfState().haveUserPurchaseRequest()) {
        const auto userPurchaseRequest = worldState_.selfState().getUserPurchaseRequest();
        // User purchased something, we saved this so that we can get the NPC's global Id
        if (worldState_.entityTracker().trackingEntity(userPurchaseRequest.globalId)) {
          auto object = worldState_.entityTracker().getEntity(userPurchaseRequest.globalId);
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
              worldState_.selfState().inventory.addItem(movement.destSlots[0], item);
              eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlots[0]));
            } else {
              // Multiple destination slots, must be unstackable items like equipment
              for (auto destSlot : movement.destSlots) {
                auto item = helpers::createItemFromScrap(itemInfo, itemRef);
                worldState_.selfState().inventory.addItem(destSlot, item);
                eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
              }
            }
          }
        }
        worldState_.selfState().resetUserPurchaseRequest();
      } else {
        LOG() << "kBuyItem but we dont have the data from the client packet\n";
        // TODO: Introduce unknown item concept?
      }
    } else if (movement.type == packet::enums::ItemMovementType::kSellItem) {
      if (worldState_.selfState().inventory.hasItem(movement.srcSlot)) {
        bool soldEntireStack = true;
        auto item = worldState_.selfState().inventory.getItem(movement.srcSlot);
        storage::ItemExpendable *itemExpendable;
        if ((itemExpendable = dynamic_cast<storage::ItemExpendable*>(item)) != nullptr) {
          if (itemExpendable->quantity != movement.quantity) {
            LOG() << "Sold only some of this item " << itemExpendable->quantity << " -> " << itemExpendable->quantity-movement.quantity << '\n';
            soldEntireStack = false;
            itemExpendable->quantity -= movement.quantity;
            auto clonedItem = storage::cloneItem(item);
            dynamic_cast<storage::ItemExpendable*>(clonedItem.get())->quantity = movement.quantity;
            worldState_.selfState().buybackQueue.addItem(clonedItem);
          }
        }
        if (soldEntireStack) {
          auto item = worldState_.selfState().inventory.withdrawItem(movement.srcSlot);
          worldState_.selfState().buybackQueue.addItem(item);
        }
        eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, std::nullopt));
      } else {
        LOG() << "Sold an item from a slot that we didnt have item data for\n";
      }
    } else if (movement.type == packet::enums::ItemMovementType::kBuyback) {
      if (worldState_.selfState().buybackQueue.hasItem(movement.srcSlot)) {
        if (!worldState_.selfState().inventory.hasItem(movement.destSlot)) {
          const auto itemPtr = worldState_.selfState().buybackQueue.getItem(movement.srcSlot);
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
                worldState_.selfState().inventory.addItem(movement.destSlot, clonedItem);
                eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
              }
            }
          }
          if (boughtBackAll) {
            worldState_.selfState().inventory.addItem(movement.destSlot, worldState_.selfState().buybackQueue.withdrawItem(movement.srcSlot));
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
        addItemToInventory(worldState_.selfState().inventory, movement.newItem, movement.destSlot);
      }
      // This would be a good time to try to use a pill, potion, return scroll, etc.
    } else if (movement.type == packet::enums::ItemMovementType::kDropItem) {
      removeItemFromInventory(movement.srcSlot);
    } else if (movement.type == packet::enums::ItemMovementType::kAddItemByServer) {
      addItemToInventory(worldState_.selfState().inventory, movement.newItem, movement.destSlot);
    } else if (movement.type == packet::enums::ItemMovementType::kRemoveItemByServer) {
      removeItemFromInventory(movement.srcSlot);
    } else if (movement.type == packet::enums::ItemMovementType::kDropGold) {
      // Another packet, ServerAgentEntityUpdatePoints, contains character gold update information
    } else if (movement.type == packet::enums::ItemMovementType::kChestWithdrawGold) {
      worldState_.selfState().setStorageGold(worldState_.selfState().getStorageGold() - movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kChestDepositGold) {
      worldState_.selfState().setStorageGold(worldState_.selfState().getStorageGold() + movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestDepositGold) {
      worldState_.selfState().setGuildStorageGold(worldState_.selfState().getGuildStorageGold() - movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kGuildChestWithdrawGold) {
      worldState_.selfState().setGuildStorageGold(worldState_.selfState().getGuildStorageGold() + movement.goldAmount);
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemAvatarToInventory) {
      worldState_.selfState().inventory.addItem(movement.destSlot, worldState_.selfState().avatarInventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::AvatarInventoryUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemInventoryToAvatar) {
      worldState_.selfState().avatarInventory.addItem(movement.destSlot, worldState_.selfState().inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::AvatarInventoryUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemCosToInventory) {
      auto &cosInventory = worldState_.selfState().getCosInventory(movement.globalId);
      worldState_.selfState().inventory.addItem(movement.destSlot, cosInventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::CosInventoryUpdated>(movement.globalId, movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kMoveItemInventoryToCos) {
      auto &cosInventory = worldState_.selfState().getCosInventory(movement.globalId);
      cosInventory.addItem(movement.destSlot, worldState_.selfState().inventory.withdrawItem(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::InventoryUpdated>(movement.srcSlot, std::nullopt));
      eventBroker_.publishEvent(std::make_unique<event::CosInventoryUpdated>(movement.globalId, std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kUpdateSlotsInventoryCos) {
      auto &cosInventory = worldState_.selfState().getCosInventory(movement.globalId);
      cosInventory.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      eventBroker_.publishEvent(std::make_unique<event::CosInventoryUpdated>(movement.globalId, movement.srcSlot, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kPickItemCos) {
      auto &cosInventory = worldState_.selfState().getCosInventory(movement.globalId);
      addItemToInventory(cosInventory, movement.newItem, movement.destSlot);
      eventBroker_.publishEvent(std::make_unique<event::CosInventoryUpdated>(movement.globalId, std::nullopt, movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kPickItemByOther) {
      // Always is our COS picking gold. Gold update packet updates our state. We dont need to handle this
    } else {
      LOG() << "Unknown item movement type: " << static_cast<int>(movement.type) << std::endl;
    }
  }
}

void PacketProcessor::serverAgentEntityGroupSpawnDataReceived(const packet::parsing::ServerAgentEntityGroupSpawnData &packet) const {
  if (packet.groupSpawnType() == packet::enums::GroupSpawnType::kSpawn) {
    for (auto entity : packet.entities()) {
      if (entity) {
        entitySpawned(entity);
      } else {
        LOG() << "Received null entity from group spawn" << std::endl;
      }
    }
  } else {
    for (auto globalId : packet.despawnGlobalIds()) {
      entityDespawned(globalId);
    }
  }
}

void PacketProcessor::serverAgentEntitySpawnReceived(const packet::parsing::ServerAgentEntitySpawn &packet) const {
  if (packet.entity()) {
    entitySpawned(packet.entity());
  } else {
    LOG() << "Received null entity from spawn" << std::endl;
  }
}

void PacketProcessor::serverAgentEntityDespawnReceived(const packet::parsing::ServerAgentEntityDespawn &packet) const {
  entityDespawned(packet.globalId());
}

void PacketProcessor::entitySpawned(std::shared_ptr<entity::Entity> entity) const {
  worldState_.entityTracker().trackEntity(entity);
  eventBroker_.publishEvent(std::make_unique<event::EntitySpawned>(entity->globalId));

  // Check if the entity spawned in as already moving
  auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(entity.get());
  if (mobileEntity == nullptr) {
    // Non-mobile, nothing to do
    return;
  }
  if (mobileEntity->moving()) {
    if (mobileEntity->destinationPosition) {
      // Entity spawned and is moving to a destination
      mobileEntity->setMovingToDestination(mobileEntity->position(), *mobileEntity->destinationPosition, eventBroker_);
    } else {
      mobileEntity->setMovingTowardAngle(mobileEntity->position(), mobileEntity->angle(), eventBroker_);
    }
  }
}

void PacketProcessor::entityDespawned(sro::scalar_types::EntityGlobalId globalId) const {
  if (!worldState_.entityTracker().trackingEntity(globalId)) {
    // TODO: Once eventzones are handled, this check can be removed;
    //  getEntity will throw
    LOG() << "Entity despawned, but we're not tracking it" << std::endl;
    return;
  }
  // Before destroying an entity, see if we have a running movement timer to cancel
  auto *entity = worldState_.entityTracker().getEntity(globalId);
  auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(entity);
  if (mobileEntity != nullptr) {
    // Is a mobile entity
    mobileEntity->cancelEvents(eventBroker_);
  }

  // Destroy entity
  worldState_.entityTracker().stopTrackingEntity(globalId);
  eventBroker_.publishEvent(std::make_unique<event::EntityDespawned>(globalId));
}

// ============================================================================================================================
// ============================================================Misc============================================================
// ============================================================================================================================

void PacketProcessor::serverAgentDeselectResponseReceived(const packet::parsing::ServerAgentActionDeselectResponse &packet) const {
  if (packet.result() == 1) {
    // Successfully deselected
    // If there is a talk dialog, and we have an npc selected, it will take 2 deselects to close both dialogs
    //  First, the talk dialog is closed
    if (worldState_.selfState().talkingGidAndOption) {
      // This closes the talk dialog
      worldState_.selfState().talkingGidAndOption.reset();
      eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kEntityDeselected));
    } else {
      //  The entity is deselected
      if (worldState_.selfState().selectedEntity) {
        worldState_.selfState().selectedEntity.reset();
        eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kEntityDeselected));
      } else {
        LOG() << "Weird, we didnt have anything selected\n";
      }
    }
  } else {
    LOG() << "Deselection failed" << std::endl;
  }
}

void PacketProcessor::serverAgentSelectResponseReceived(const packet::parsing::ServerAgentActionSelectResponse &packet) const {
  if (packet.result() != 1) {
    LOG() << "Selection failed" << std::endl;
    return;
  }

  // Successfully selected
  // It is possible that we already have something selected. We will just overwrite it
  worldState_.selfState().selectedEntity = packet.globalId();
  auto *entity = worldState_.entityTracker().getEntity(packet.globalId());
  if (entity == nullptr) {
    throw std::runtime_error("Selected an entity which we are not tracking");
  }
  if (auto *monster = dynamic_cast<entity::Monster*>(entity)) {
    // Selected a monster
    if (flags::isSet(packet.vitalInfoMask(), packet::enums::VitalInfoFlag::kVitalInfoHp)) {
      // Received monster's current HP
      LOG() << "Selected monster. Setting HP" << std::endl;
      monster->setCurrentHp(packet.hp(), eventBroker_);
    }
  }
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kEntitySelected));
}

void PacketProcessor::serverAgentTalkResponseReceived(const packet::parsing::ServerAgentActionTalkResponse &packet) const {
  if (packet.result() == 1) {
    // Successfully talking to an npc
    if (worldState_.selfState().pendingTalkGid) {
      // We were waiting for this response
      worldState_.selfState().talkingGidAndOption = std::make_pair(*worldState_.selfState().pendingTalkGid, packet.talkOption());
      worldState_.selfState().pendingTalkGid.reset();
      eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kNpcTalkStart));
    } else {
      LOG() << "Weird, we werent expecting to be talking to anything. As a result, we dont know what we're talking to" << std::endl;
    }
  } else {
    LOG() << "Failed to talk to NPC" << std::endl;
  }
}

void PacketProcessor::serverAgentInventoryRepairResponseReceived(const packet::parsing::ServerAgentInventoryRepairResponse &packet) const {
  if (packet.successful()) {
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kRepairSuccessful));
  } else {
    LOG() << "Repairing item(s) failed! Error code: " << packet.errorCode() << std::endl;
  }
}

void PacketProcessor::serverAgentInventoryUpdateDurabilityReceived(const packet::parsing::ServerAgentInventoryUpdateDurability &packet) const {
  if (!worldState_.selfState().inventory.hasItem(packet.slotIndex())) {
    throw std::runtime_error("Recieved durability update for inventory slot where no item exists");
  }
  auto *item = worldState_.selfState().inventory.getItem(packet.slotIndex());
  if (item == nullptr) {
    throw std::runtime_error("Recieved durability update for inventory item which is null");
  }
  auto *itemAsEquip = dynamic_cast<storage::ItemEquipment*>(item);
  if (itemAsEquip == nullptr) {
    throw std::runtime_error("Recieved durability update for inventory item which is not a piece of equipment");
  }
  // Update item's durability
  itemAsEquip->durability = packet.durability();
}

void PacketProcessor::serverAgentInventoryUpdateItemReceived(const packet::parsing::ServerAgentInventoryUpdateItem &packet) const {
  if (!worldState_.selfState().inventory.hasItem(packet.slotIndex())) {
    throw std::runtime_error("Recieved item update for inventory slot where no item exists");
  }
  auto *item = worldState_.selfState().inventory.getItem(packet.slotIndex());
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
}

// void PacketProcessor::clientAgentActionDeselectRequestReceived(const packet::parsing::ClientAgentActionDeselectRequest &packet) const {
// }

// void PacketProcessor::clientAgentActionSelectRequestReceived(const packet::parsing::ClientAgentActionSelectRequest &packet) const {
// }

void PacketProcessor::clientAgentActionTalkRequestReceived(const packet::parsing::ClientAgentActionTalkRequest &packet) const {
  if (worldState_.selfState().pendingTalkGid) {
    LOG() << "Weird, we're already waiting on a response from the server to talk to someone\n";
  } else {
    worldState_.selfState().pendingTalkGid = packet.gId();
  }
}

void PacketProcessor::serverAgentEntityUpdatePointsReceived(const packet::parsing::ServerAgentEntityUpdatePoints &packet) const {
  if (packet.updatePointsType() == packet::enums::UpdatePointsType::kGold) {
    worldState_.selfState().setGold(packet.gold());
  } else if (packet.updatePointsType() == packet::enums::UpdatePointsType::kSp) {
    worldState_.selfState().setSkillPoints(packet.skillPoints());
  }
}

void PacketProcessor::serverAgentEntityUpdateExperienceReceived(const packet::parsing::ServerAgentEntityUpdateExperience &packet) const {
  const constexpr int kSpExperienceRequired{400}; // TODO: Move to a central location
  const auto newExperience = worldState_.selfState().getCurrentExperience() + packet.gainedExperiencePoints();
  const auto newSpExperience = (worldState_.selfState().getCurrentSpExperience() + packet.gainedSpExperiencePoints()) % kSpExperienceRequired;
  worldState_.selfState().setCurrentExpAndSpExp(newExperience, newSpExperience);
}

void PacketProcessor::serverAgentGuildStorageDataReceived(const packet::parsing::ServerAgentGuildStorageData &packet) const {
  worldState_.selfState().setGuildStorageGold(packet.gold());
  helpers::initializeInventory(worldState_.selfState().guildStorage, packet.storageSize(), packet.storageItemMap());
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kGuildStorageInitialized));
}

void PacketProcessor::clientAgentActionCommandRequestReceived(const packet::parsing::ClientAgentActionCommandRequest &packet) const {
  LOG() << "Client command action received: \"" << packet.actionCommand() << "\"";
  if (packet.actionCommand().commandType == packet::enums::CommandType::kExecute &&
      packet.actionCommand().actionType == packet::enums::ActionType::kCast) {
    const auto skillName = gameData_.getSkillNameIfExists(packet.actionCommand().refSkillId);
    if (skillName) {
      std::cout << ", skillName: \"" << *skillName << "\"";
    }
  }
  std::cout << std::endl;
  worldState_.selfState().pendingCommandQueue.push_back(packet.actionCommand());
}

void PacketProcessor::serverAgentActionCommandResponseReceived(const packet::parsing::ServerAgentActionCommandResponse &packet) const {
  LOG() << "Received command response" << std::endl;
  std::cout << "=========================================== " << packet.actionState() << "============================================" << std::endl;
  std::cout << "Pending command Queue:" << (worldState_.selfState().pendingCommandQueue.empty() ? " <empty>" : "") << std::endl;
  for (const auto &c : worldState_.selfState().pendingCommandQueue) {
    std::cout << "  " << c << std::endl;
  }
  std::cout << "Accepted command Queue:" << (worldState_.selfState().acceptedCommandQueue.empty() ? " <empty>" : "") << std::endl;
  for (const auto &c : worldState_.selfState().acceptedCommandQueue) {
    std::cout << "  " << c.command << std::endl;
  }
  std::cout << "----------------" << std::endl;

  if (packet.actionState() == packet::enums::ActionState::kQueued) {
    if (worldState_.selfState().pendingCommandQueue.empty()) {
      throw std::runtime_error("Command queued, but pending command list is empty");
    }
    std::cout << "Command queued: \"" << worldState_.selfState().pendingCommandQueue.front() << "\"" << std::endl;
    worldState_.selfState().acceptedCommandQueue.emplace_back(worldState_.selfState().pendingCommandQueue.front());
    worldState_.selfState().pendingCommandQueue.erase(worldState_.selfState().pendingCommandQueue.begin());
  } else if (packet.actionState() == packet::enums::ActionState::kError) {
    // 16388 happens when the skill is on cooldown
    if (worldState_.selfState().pendingCommandQueue.empty()) {
      throw std::runtime_error("Command error, but pending command list is empty");
    }
    // Error seems to always refer to the most recent
    const auto failedCommand = worldState_.selfState().pendingCommandQueue.front();
    std::cout << "Command error " << packet.errorCode() << ": " << failedCommand << std::endl;
    eventBroker_.publishEvent(std::make_unique<event::CommandError>(failedCommand));
    worldState_.selfState().pendingCommandQueue.erase(worldState_.selfState().pendingCommandQueue.begin());
  } else /*if (packet.actionState() == packet::enums::ActionState::kEnd)*/ {
    // It seems like if a skill is completed without interruption, this end will come after the SkillEnd packet
    // If a skill is interrupted, this end will come BEFORE the SkillEnd packet
    if (worldState_.selfState().acceptedCommandQueue.empty()) {
      std::cout << "WARNING: Command ended, but we had no accepted command" << std::endl;
      if (!worldState_.selfState().pendingCommandQueue.empty()) {
        std::cout << " Pending command queue is not empty though, maybe we ought to pop that?" << std::endl;
        if (worldState_.selfState().pendingCommandQueue.front().commandType == packet::enums::CommandType::kCancel) {
          std::cout << "  Action is a cancel, popping" << std::endl;
          // TODO: I am not confident in the assumption that this means that we delete the first item in the pending queue
          worldState_.selfState().pendingCommandQueue.erase(worldState_.selfState().pendingCommandQueue.begin());
        }
      }
    } else {
      // We arent told which command ended, we just assume it was the most recent
      std::cout << "Command ended: \"" << worldState_.selfState().acceptedCommandQueue.front().command << "\"" << std::endl;
      if (worldState_.selfState().acceptedCommandQueue.front().command.commandType == packet::enums::CommandType::kExecute &&
          worldState_.selfState().acceptedCommandQueue.front().command.actionType == packet::enums::ActionType::kCast &&
          !worldState_.selfState().acceptedCommandQueue.front().wasExecuted) {
        std::cout << "  this command(skill) was never executed!!" << std::endl;
        eventBroker_.publishEvent(std::make_unique<event::CommandError>(worldState_.selfState().acceptedCommandQueue.front().command));
        // TODO: Maybe this should be a different event than "CommandError"?
      }
      worldState_.selfState().acceptedCommandQueue.erase(worldState_.selfState().acceptedCommandQueue.begin());
    }
  }

  std::cout << "----------------" << std::endl;
  std::cout << "Pending command Queue:" << (worldState_.selfState().pendingCommandQueue.empty() ? " <empty>" : "") << std::endl;
  for (const auto &c : worldState_.selfState().pendingCommandQueue) {
    std::cout << "  " << c << std::endl;
  }
  std::cout << "Accepted command Queue:" << (worldState_.selfState().acceptedCommandQueue.empty() ? " <empty>" : "") << std::endl;
  for (const auto &c : worldState_.selfState().acceptedCommandQueue) {
    std::cout << "  " << c.command << std::endl;
  }
  std::cout << "==========================================================================================" << std::endl;
}

// Skill Notes:
/*
ActionCastingTime is the amount of time it takes for the skill to "end" after it began
ActionActionDuration is the amount of time it takes for the skill to cast before the character is free
ActionReuseDelay is the skill's cooldown
*/

void PacketProcessor::serverAgentSkillBeginReceived(const packet::parsing::ServerAgentSkillBegin &packet) const {
  auto getRootSkillRefId = [&, this](auto refId) {
    const auto &skillData = gameData_.skillData().getSkillById(refId);
    if ()
  };
  if (packet.result() == 2) {
    // Error
    // LOG() << "Skill unsuccessful, err " << packet.errorCode() << std::endl;
    if (packet.casterGlobalId()) {
      // Which skill is this? It must be the first item in the pendingCommandQueue
      if (!worldState_.selfState().pendingCommandQueue.empty()) {
        const auto nextCommand = worldState_.selfState().pendingCommandQueue.front();
        if (nextCommand.commandType == packet::enums::CommandType::kExecute) {
          if (nextCommand.actionType == packet::enums::ActionType::kCast) {
            const auto skillRefId = nextCommand.refSkillId;
            // LOG() << "Our skill failed (" << skillRefId << ')' << std::endl;
            eventBroker_.publishEvent(std::make_unique<event::OurSkillFailed>(skillRefId, packet.errorCode()));
          } else {
            // LOG() << "Out skill failed, but the most recent pending command isnt a \"Cast\"" << std::endl;
          }
        } else {
          // LOG() << "Our skill failed, but the most recent pending command isnt an \"Execute\"" << std::endl;
        }
      } else {
        // LOG() << "Our skill failed, but we dont know which one!" << std::endl;
      }
    }
    return;
  }

  // Update world state based on action
  handleSkillAction(packet.action(), packet.casterGlobalId());

  // Do some skill tracking work
  if (packet.casterGlobalId() == worldState_.selfState().globalId) {
    // Only tracking our own skills for now
    eventBroker_.publishEvent(std::make_unique<event::SkillBegan>(worldState_.selfState().globalId, packet.refSkillId()));
    const auto &skillData = gameData_.skillData().getSkillById(packet.refSkillId());
    const auto rootSkillRefId = gameData_.skillData().getRootSkillRefId(packet.refSkillId());
    const bool isRootSkill = rootSkillRefId == packet.refSkillId();
    const bool skillIsCommonAttack = skillData.basicCode.find("BASE") != std::string::npos; // TODO: Find a more robust way to check
    auto skillName = [&]() -> std::string {
      if (skillIsCommonAttack) {
        return "Common";
      }
      const auto skillName = gameData_.textItemAndSkillData().getSkillNameIfExists(skillData.uiSkillName);
      if (!skillName) {
        return "UNKNOWN";
      }
      return *skillName;
    }();
    // LOG() << "SkillBegin \"" << skillName << "\" (" << packet.refSkillId() << ") with casting time: " << skillData.actionCastingTime << ", action duration: " << skillData.actionActionDuration << ", and reuse delay: " << skillData.actionReuseDelay << std::endl;
    if (!isRootSkill) {
      // LOG() << "  Skill " << packet.refSkillId() << "'s root is " << rootSkillRefId << std::endl;
    }
    // We expect that were is at least one accepted command in the queue
    const bool isFinalPieceOfChain = gameData_.skillData().getSkillById(packet.refSkillId()).basicChainCode == 0;
    if (!worldState_.selfState().acceptedCommandQueue.empty()) {
      // Try to find the index of this skill in the accepted command queue
      std::optional<size_t> indexOfOurSkill;
      int indexCounter=0;
      for (const auto &acceptedCommand : worldState_.selfState().acceptedCommandQueue) {
        if (acceptedCommand.command.refSkillId == rootSkillRefId) {
          indexOfOurSkill = indexCounter;
          break;
        }
        ++indexCounter;
      }
      // TODO: Common attacks will not be found, does that matter? That means that no skill cooldown will be created for one
      if (!indexOfOurSkill) {
        std::cout << "Couldn't find our skill in the accepted command queue" << std::endl;
        // This happens for common attacks
        if (skillIsCommonAttack && !(worldState_.selfState().acceptedCommandQueue.front().command.commandType == packet::enums::CommandType::kExecute && worldState_.selfState().acceptedCommandQueue.front().command.actionType == packet::enums::ActionType::kAttack)) {
          // First command is not a common attack
          // LOG() << "First command in the queue isnt a common attack!" << std::endl;
        }
      } else if (*indexOfOurSkill != 0) {
        // Remove all commands before this one in the queue, those have probably been discarded by the server
        // LOG() << "Our skill is not the first in the accepted command queue" << std::endl;
        for (int i=0; i<*indexOfOurSkill; ++i) {
          // LOG() << "Command #" << i << " (" << worldState_.selfState().acceptedCommandQueue.at(i).command << ") skipped" << std::endl;
          // TODO: Should we publish an event that command has been skipped?
        }
        worldState_.selfState().acceptedCommandQueue.erase(worldState_.selfState().acceptedCommandQueue.begin(), worldState_.selfState().acceptedCommandQueue.begin() + *indexOfOurSkill);
        indexOfOurSkill = 0;
      }
      if (indexOfOurSkill) {
        // We cast this skill
        // Assumption (90% certain of): A skill always has a begin, but might not have an end
        //  Marking this skill as executed here is sufficient
        worldState_.selfState().acceptedCommandQueue.at(*indexOfOurSkill).wasExecuted = true;
        if (isRootSkill) {
          // Set a timer for when the skill cooldown ends
          worldState_.selfState().skillsOnCooldown.emplace(packet.refSkillId());
          eventBroker_.publishDelayedEvent(std::make_unique<event::SkillCooldownEnded>(packet.refSkillId()), std::chrono::milliseconds(skillData.actionReuseDelay));
        } else if (skillData.actionReuseDelay != 0) {
          throw std::runtime_error("Cast a non-root-skill with a cooldown!");
        }
      }
    } else {
      // This happens when we spawn in with a speed scroll
      // LOG() << "WARNING: accepted command queue empty" << std::endl;
    }
    if (skillData.actionCastingTime == 0) {
      // Skill takes no time to cast, I think there wont be an "end" coming from the server
      // TODO: Some skills have 0 casting time, but an "end" will come from the server
      //  Iron Skin
      //  Mana Skin
      //  Holy Spell
      //  Crossbow Extreme

      if (isFinalPieceOfChain) {
        eventBroker_.publishEvent(std::make_unique<event::SkillEnded>(worldState_.selfState().globalId, packet.refSkillId()));
      } else {
        // LOG() << "Isn't final piece of chain, not going to send an \"end\" yet" << std::endl;
      }
    } else {
      // We cast this skill, save the cast ID so that we can reference it later on SkillEnd
      worldState_.selfState().skillCastIdMap.emplace(std::piecewise_construct, std::forward_as_tuple(packet.castId()), std::forward_as_tuple(packet.casterGlobalId(), packet.refSkillId()));
      if (worldState_.selfState().skillCastIdMap.size() > 1) {
        // LOG() << "  Skill casts tracked: [ ";
        // for (const auto &i : worldState_.selfState().skillCastIdMap) {
        //   std::cout << i.first << ", ";
        // }
        // std::cout << "]" << std::endl;
      }
    }
  } else {
    // Caster is not us
    if (auto *monster = dynamic_cast<entity::Monster*>(worldState_.entityTracker().getEntity(packet.casterGlobalId()))) {
      // Caster is a monster, track who the monster is targetting
      monster->targetGlobalId = packet.targetGlobalId();
    }
  }

  if (!gameData_.skillData().haveSkillWithId(packet.refSkillId())) {
    throw std::runtime_error("Cast a skill which we dont have data for");
  }
  const auto &skill = gameData_.skillData().getSkillById(packet.refSkillId());
  switch (skill.basicActivity) {
    case 0:
      // Seems to be passives
      // LOG() << "Cast a skill with basic activity == 0" << std::endl;
      break;
    case 1:
      // Dont stop while running. Can be cast while something else is being case
      break;
    case 2:
      {
        // Will stop you if you're running
        entity::MobileEntity &casterAsMobileEntity = worldState_.getEntity<entity::MobileEntity>(packet.casterGlobalId());
        if (casterAsMobileEntity.moving()) {
          casterAsMobileEntity.setStationaryAtPosition(casterAsMobileEntity.position(), eventBroker_);
        }
        break;
      }
    default:
      throw std::runtime_error("Cast a skill with unknown basic activity == "+std::to_string(skill.basicActivity));
      break;
  }
}

void PacketProcessor::serverAgentSkillEndReceived(const packet::parsing::ServerAgentSkillEnd &packet) const {
  // LOG() << "Skill end packet received" << std::endl;
  if (packet.result() == 2) {
    // Not successful?
    return;
  }
  handleSkillAction(packet.action());

  // Have we tracked this skill?
  if (auto it = worldState_.selfState().skillCastIdMap.find(packet.castId()); it!=worldState_.selfState().skillCastIdMap.end()) {
    auto &skillInfo = it->second;
    const auto thisSkillId = skillInfo.skillRefId;
    if (skillInfo.casterGlobalId == worldState_.selfState().globalId) {
      // We cast this skill
      const auto &skillData = gameData_.skillData().getSkillById(thisSkillId);
      const auto maybeSkillName = gameData_.textItemAndSkillData().getSkillNameIfExists(skillData.uiSkillName);
      // LOG() << "  Is our \"" << (maybeSkillName ? *maybeSkillName : "UNKNOWN") << "\" end" << std::endl;
      bool doneWithSkill{true};
      if (skillData.basicChainCode != 0) {
        // There are more pieces to this skill
        skillInfo.skillRefId = skillData.basicChainCode;
        const auto &nextSkillPiece = gameData_.skillData().getSkillById(skillInfo.skillRefId);
        if (nextSkillPiece.actionCastingTime != 0) {
          doneWithSkill = false;
          // LOG() << "  Another piece of skill is coming, basic chain code=" << skillData.basicChainCode << ". This skill has non-zero cast time: " << nextSkillPiece.actionCastingTime << std::endl;
        } else {
          // LOG() << "  Skill has another piece, but has 0 casting time." << std::endl;
          // TODO: Check if there is ANOTHER piece coming...
        }
      }
      if (doneWithSkill) {
        // LOG() << "  Done with skill" << std::endl;
        eventBroker_.publishEvent(std::make_unique<event::SkillEnded>(worldState_.selfState().globalId, thisSkillId));
        // Remove it from the map
        worldState_.selfState().skillCastIdMap.erase(it);
      }
    } else {
      // LOG() << "  Is NOT our skill end" << std::endl;
    }
  } else {
    // LOG() << "  Untracked cast" << std::endl;
  }
}

void PacketProcessor::handleSkillAction(const packet::structures::SkillAction &action, std::optional<sro::scalar_types::EntityGlobalId> casterGlobalId) const {
  if (casterGlobalId &&
     (flags::isSet(action.actionFlag, packet::enums::ActionFlag::kTeleport) ||
      flags::isSet(action.actionFlag, packet::enums::ActionFlag::kSprint))) {
    // Entity teleported or sprinted to a new position. Sprints are actually teleports on the server side, even though the skill actually has a duration. The duration is only used for animation
    entity::MobileEntity &casterAsMobileEntity = worldState_.getEntity<entity::MobileEntity>(*casterGlobalId);
    casterAsMobileEntity.setStationaryAtPosition(action.position, eventBroker_);
  }

  for (const auto &hitObject : action.hitObjects) {
    for (const auto &hitResult : hitObject.hits) {
      entity::Entity &targetEntity = worldState_.getEntity<entity::Entity>(hitObject.targetGlobalId);
      if (auto *character = dynamic_cast<entity::Character*>(&targetEntity)) {
        if (flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKill)) {
          // Effectively killed it, but I don't know if it makes sense to change the life state right now
          character->setCurrentHp(0, eventBroker_);
        } else {
          if (flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKnockdown)) {
            LOG() << "Entity has been knocked down" << std::endl;
            // TODO: Update entity state, publish knocked down event(?), publish delayed stood up event
          }
          if (hitObject.targetGlobalId == worldState_.selfState().globalId) {
            if (flags::isSet(hitResult.damageFlag, packet::enums::DamageFlag::kEffect) && hitResult.effect != 0) {
              // Applied an effect to us
              LOG() << "We have been hit with effect: " << static_cast<packet::enums::AbnormalStateFlag>(hitResult.effect) << std::endl;
            }
          }
          if (character->knowCurrentHp()) {
            // Can only update the character's hp if we know what it currently is
            if (hitResult.damage >= character->currentHp()) {
              character->setCurrentHp(0, eventBroker_);
            } else {
              character->setCurrentHp(character->currentHp() - hitResult.damage, eventBroker_);
            }
          }
        }
      }
      if (flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKnockback) ||
          flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKnockdown)) {
        if (auto *targetAsMobileEntity = dynamic_cast<entity::MobileEntity*>(&targetEntity)) {
          targetAsMobileEntity->setStationaryAtPosition(hitResult.position, eventBroker_);
          if (hitObject.targetGlobalId == worldState_.selfState().globalId) {
            // We are the target
            bool knockedBackOrKnockedDown{false};
            if (flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKnockback)) {
              constexpr const int kKnockbackStunDuration{2000};
              LOG() << "We were knocked back " << static_cast<int>(hitResult.hitResultFlag) << ", sending stun delayed event " << kKnockbackStunDuration << "ms" << std::endl;
              worldState_.selfState().stunnedFromKnockback = true;
              knockedBackOrKnockedDown = true;
              // Publish knocked back event
              eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kKnockedBack));
              // Publish delayed knocked back stun completed event
              eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kKnockbackStunEnded), std::chrono::milliseconds(kKnockbackStunDuration));
            } else if (flags::isSet(hitResult.hitResultFlag, packet::enums::HitResult::kKnockdown)) {
              constexpr const int kKnockdownStunDuration{6000};
              LOG() << "We were knocked down " << static_cast<int>(hitResult.hitResultFlag) << ", sending stun delayed event " << kKnockdownStunDuration << "ms" << std::endl;
              worldState_.selfState().stunnedFromKnockdown = true;
              knockedBackOrKnockedDown = true;
              // Publish knocked down event
              eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kKnockedDown));
              // Publish delayed knocked down stun completed event
              eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kKnockdownStunEnded), std::chrono::milliseconds(kKnockdownStunDuration));
            }
            if (knockedBackOrKnockedDown) {
              // Whatever commands we had queued should probably be cleared
              // Whatever skills were casting should probably be cancelled
              handleKnockedBackOrKnockedDown();
            }
          }
        } else {
          throw std::runtime_error("Knockback/knockdown on non-mobile entity");
        }
      }
    }
  }
}

void PacketProcessor::handleKnockedBackOrKnockedDown() const {
  // When knocked back or knocked down, all accepted commands are not going to execute
  worldState_.selfState().acceptedCommandQueue.clear();
  // It doesn't make sense to remove all pending commands as those have not even been acknowledge by the server yet
  //  The server will likely respond with an error response for them
  if (!worldState_.selfState().skillCastIdMap.empty()) {
    LOG() << "KB/KD with active casts: ";
    for (const auto &i : worldState_.selfState().skillCastIdMap) {
      std::cout << i.first << ", ";
    }
    std::cout << ". Clearing" << std::endl;
    // TODO: Verify if it makes sense to clear this
    //  It certainly seems like all started casts will be interrupted
    worldState_.selfState().skillCastIdMap.clear();
  }
}

void PacketProcessor::serverAgentBuffAddReceived(const packet::parsing::ServerAgentBuffAdd &packet) const {
  if (packet.activeBuffToken() == 0) {
    // No buff remove will be received when this expires
    //  Seems to be only for debuffs
    return;
  }
  if (packet.globalId() == worldState_.selfState().globalId) {
    const auto skillName = gameData_.getSkillNameIfExists(packet.skillRefId());
    LOG() << "Buff \"" << (skillName ? *skillName : "UNKNOWN") << "\" added to us with tokenId: " << packet.activeBuffToken() << std::endl;
  }
  worldState_.addBuff(packet.globalId(), packet.skillRefId(), packet.activeBuffToken());
}

void PacketProcessor::serverAgentBuffRemoveReceived(const packet::parsing::ServerAgentBuffRemove &packet) const {
  LOG() << "Buff remove received. Buffs: [ ";
  for (const auto &tokenId : packet.tokens()) {
    std::cout << tokenId << ", ";
  }
  std::cout << "]" << std::endl;
  worldState_.removeBuffs(packet.tokens());
}

void PacketProcessor::clientAgentInventoryItemUseRequestReceived(const packet::parsing::ClientAgentInventoryItemUseRequest &packet) const {
  worldState_.selfState().pushItemToUsedItemQueue(packet.inventoryIndex(), packet.itemTypeId());
}