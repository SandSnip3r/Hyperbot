#include "logging.hpp"
#include "userInterface.hpp"

#include "pk2/gameData.hpp"
#include "state/worldState.hpp"

#include "ui-proto/request.pb.h"

namespace {

proto::broadcast::LifeState lifeStateToProto(sro::entity::LifeState lifeState) {
  switch (lifeState) {
    case sro::entity::LifeState::kEmbryo:
      return proto::broadcast::LifeState::kEmbryo;
    case sro::entity::LifeState::kAlive:
      return proto::broadcast::LifeState::kAlive;
    case sro::entity::LifeState::kDead:
      return proto::broadcast::LifeState::kDead;
    case sro::entity::LifeState::kGone:
      return proto::broadcast::LifeState::kGone;
    default:
      throw std::runtime_error("Unknown lifestate");
  }
}

} // anonymous namespace

namespace ui {

UserInterface::UserInterface(const pk2::GameData &gameData, broker::EventBroker &eventBroker) : gameData_(gameData), eventBroker_(eventBroker) {
  //
}

UserInterface::~UserInterface() {
  thr_.join();
}

void UserInterface::initialize() {
  subscribeToEvents();
}

void UserInterface::setWorldState(const state::WorldState &worldState) {
  worldState_ = &worldState;
}

void UserInterface::runAsync() {
  // Set up publisher
  publisher_.bind("tcp://*:5556");

  // Run the request receiver in another thread
  thr_ = std::thread(&UserInterface::run, this);
}

void UserInterface::broadcastLaunch() {
  // Broadcast that we just started. If there's a UI out there, this lets it know that we've freshly started the bot and it can reset its state.
  proto::broadcast::BroadcastMessage broadcastMessage;
  broadcastMessage.mutable_launch();
  broadcast(broadcastMessage);
}

void UserInterface::subscribeToEvents() {
  auto eventHandleFunction = std::bind(&UserInterface::handleEvent, this, std::placeholders::_1);

  eventBroker_.subscribeToEvent(event::EventCode::kSpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCosSpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntitySpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityDespawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityLifeStateChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEnteredNewRegion, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityHpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMaxHpMpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventoryGoldUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStorageGoldUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kGuildStorageGoldUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterSkillPointsUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterExperienceUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kTrainingAreaSet, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kTrainingAreaReset, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateMachineCreated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateMachineDestroyed, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementBegan, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityPositionUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityNotMovingAngleChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStorageInitialized, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kGuildStorageInitialized, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventoryUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kAvatarInventoryUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCosInventoryUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStorageUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kGuildStorageUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kWalkingPathUpdated, eventHandleFunction);
}

void UserInterface::handleEvent(const event::Event *event) {
  std::unique_lock<std::mutex> selfStateLock(worldState_->selfState().selfMutex);

  try {
    const auto eventCode = event->eventCode;

    if (eventCode == event::EventCode::kSpawned) {
      handleSelfSpawned();
      return;
    }

    if (eventCode == event::EventCode::kCosSpawned) {
      const event::CosSpawned &castedEvent = dynamic_cast<const event::CosSpawned&>(*event);
      handleCosSpawned(castedEvent);
      return;
    }

    if (eventCode == event::EventCode::kEntitySpawned) {
      const auto &castedEvent = dynamic_cast<const event::EntitySpawned&>(*event);
      handleEntitySpawned(castedEvent);
      return;
    }

    if (eventCode == event::EventCode::kEntityDespawned) {
      const auto &castedEvent = dynamic_cast<const event::EntityDespawned&>(*event);
      broadcastEntityDespawned(castedEvent.globalId);
      return;
    }

    if (eventCode == event::EventCode::kEntityLifeStateChanged) {
      const auto &castedEvent = dynamic_cast<const event::EntityLifeStateChanged&>(*event);
      const entity::Character &characterEntity = worldState_->getEntity<entity::Character>(castedEvent.globalId);
      broadcastEntityLifeStateChanged(characterEntity.globalId, characterEntity.lifeState);
      return;
    }

    if (eventCode == event::EventCode::kEnteredNewRegion) {
      const auto pos = worldState_->selfState().position();
      const auto &regionName = gameData_.textZoneNameData().getRegionName(pos.regionId());
      broadcastRegionNameUpdate(regionName);
      return;
    }

    if (eventCode == event::EventCode::kEntityHpChanged) {
      const event::EntityHpChanged &castedEvent = dynamic_cast<const event::EntityHpChanged&>(*event);
      if (castedEvent.globalId == worldState_->selfState().globalId) {
        broadcastCharacterHpUpdate(worldState_->selfState().currentHp());
      }
      return;
    }

    if (eventCode == event::EventCode::kMpChanged) {
      broadcastCharacterMpUpdate(worldState_->selfState().currentMp());
      return;
    }

    if (eventCode == event::EventCode::kMaxHpMpChanged) {
      if (worldState_->selfState().maxHp() && worldState_->selfState().maxMp()) {
        broadcastCharacterMaxHpMpUpdate(*worldState_->selfState().maxHp(), *worldState_->selfState().maxMp());
      }
      return;
    }
    if (eventCode == event::EventCode::kInventoryGoldUpdated) {
      broadcastGoldAmountUpdate(worldState_->selfState().getGold(), proto::broadcast::ItemLocation::kCharacterInventory);
      return;
    }

    if (eventCode == event::EventCode::kStorageGoldUpdated) {
      broadcastGoldAmountUpdate(worldState_->selfState().getStorageGold(), proto::broadcast::ItemLocation::kStorage);
      return;
    }

    if (eventCode == event::EventCode::kGuildStorageGoldUpdated) {
      broadcastGoldAmountUpdate(worldState_->selfState().getGuildStorageGold(), proto::broadcast::ItemLocation::kGuildStorage);
      return;
    }

    if (eventCode == event::EventCode::kCharacterSkillPointsUpdated) {
      broadcastCharacterSpUpdate(worldState_->selfState().getSkillPoints());
      return;
    }

    if (eventCode == event::EventCode::kCharacterExperienceUpdated) {
      broadcastCharacterExperienceUpdate(worldState_->selfState().getCurrentExperience(), worldState_->selfState().getCurrentSpExperience());
      return;
    }

    if (eventCode == event::EventCode::kTrainingAreaSet) {
      if (!worldState_->selfState().trainingAreaGeometry) {
        throw std::runtime_error("Training area set, but no training geometry");
      }
      broadcastTrainingAreaSet(worldState_->selfState().trainingAreaGeometry.get());
      return;
    }

    if (eventCode == event::EventCode::kTrainingAreaReset) {
      broadcastTrainingAreaReset();
      return;
    }

    if (eventCode == event::EventCode::kStateMachineCreated) {
      const auto &castedEvent = dynamic_cast<const event::StateMachineCreated&>(*event);
      broadcastStateMachineCreated(castedEvent.stateMachineName);
      return;
    }

    if (eventCode == event::EventCode::kStateMachineDestroyed) {
      broadcastStateMachineDestroyed();
      return;
    }

    if (eventCode == event::EventCode::kEntityMovementBegan) {
      const auto &castedEvent = dynamic_cast<const event::EntityMovementBegan&>(*event);
      handleEntityMovementBegan(castedEvent);
      return;
    }

    if (eventCode == event::EventCode::kEntityMovementEnded) {
      const auto &castedEvent = dynamic_cast<const event::EntityMovementEnded&>(*event);
      handleEntityMovementEnded(castedEvent);
      return;
    }

    if (eventCode == event::EventCode::kEntityPositionUpdated) {
      const auto &castedEvent = dynamic_cast<const event::EntityPositionUpdated&>(*event);
      handleEntityPositionUpdated(castedEvent.globalId);
      return;
    }

    if (eventCode == event::EventCode::kEntityNotMovingAngleChanged) {
      const auto &castedEvent = dynamic_cast<const event::EntityNotMovingAngleChanged&>(*event);
      handleEntityNotMovingAngleChanged(castedEvent.globalId);
      return;
    }

    if (eventCode == event::EventCode::kStorageInitialized) {
      handleStorageInitialized();
      return;
    }

    if (eventCode == event::EventCode::kGuildStorageInitialized) {
      handleGuildStorageInitialized();
      return;
    }

    if (eventCode == event::EventCode::kInventoryUpdated) {
      const event::InventoryUpdated &castedEvent = dynamic_cast<const event::InventoryUpdated&>(*event);
      handleInventoryUpdated(castedEvent);
      return;
    }

    if (eventCode == event::EventCode::kAvatarInventoryUpdated) {
      const event::AvatarInventoryUpdated &castedEvent = dynamic_cast<const event::AvatarInventoryUpdated&>(*event);
      handleAvatarInventoryUpdated(castedEvent);
      return;
    }

    if (eventCode == event::EventCode::kCosInventoryUpdated) {
      const event::CosInventoryUpdated &castedEvent = dynamic_cast<const event::CosInventoryUpdated&>(*event);
      handleCosInventoryUpdated(castedEvent);
      return;
    }

    if (eventCode == event::EventCode::kStorageUpdated) {
      const event::StorageUpdated &castedEvent = dynamic_cast<const event::StorageUpdated&>(*event);
      handleStorageUpdated(castedEvent);
      return;
    }

    if (eventCode == event::EventCode::kGuildStorageUpdated) {
      const event::GuildStorageUpdated &castedEvent = dynamic_cast<const event::GuildStorageUpdated&>(*event);
      handleGuildStorageUpdated(castedEvent);
      return;
    }

    if (eventCode == event::EventCode::kWalkingPathUpdated) {
      const event::WalkingPathUpdated &castedEvent = dynamic_cast<const event::WalkingPathUpdated&>(*event);
      handleWalkingPathUpdated(castedEvent);
      return;
    }

    LOG() << "Unhandled event subscribed to. Code:" << static_cast<int>(eventCode) << '\n';
  } catch (std::exception &ex) {
    LOG() << "Error while handling event!\n  " << ex.what() << std::endl;
  }
}

void UserInterface::run() {
  // Run request receiver
  zmq::socket_t socket(context_, zmq::socket_type::rep);
  socket.bind("tcp://*:5555");
  while (1) {
    // Wait for a request
    zmq::message_t request;
    socket.recv(request, zmq::recv_flags::none);

    handleRequest(request);

    // Immediately respond with an acknowledgement
    const std::string response{"ack"};
    socket.send(zmq::buffer(response), zmq::send_flags::none);
  }
}

void UserInterface::handleRequest(const zmq::message_t &request) {
  // Parse the request
  proto::request::RequestMessage requestMsg;
  requestMsg.ParseFromArray(request.data(), request.size());
  switch (requestMsg.body_case()) {
    case proto::request::RequestMessage::BodyCase::kPacketData: {
        const proto::request::PacketToInject &packet = requestMsg.packetdata();
        const event::InjectPacket::Direction dir = (packet.direction() == proto::request::PacketToInject::kClientToServer) ? event::InjectPacket::Direction::kClientToServer : event::InjectPacket::Direction::kServerToClient;
        eventBroker_.publishEvent<event::InjectPacket>(dir, packet.opcode(), packet.data());
        break;
      }
    case proto::request::RequestMessage::BodyCase::kDoAction: {
        const proto::request::DoAction &doActionMsg = requestMsg.doaction();
        if (doActionMsg.action() == proto::request::DoAction::kStartTraining) {
          eventBroker_.publishEvent(event::EventCode::kStartTraining);
        } else if (doActionMsg.action() == proto::request::DoAction::kStopTraining) {
          eventBroker_.publishEvent(event::EventCode::kStopTraining);
        }
        break;
      }
    case proto::request::RequestMessage::BodyCase::kConfig: {
        const proto::config::Config &config = requestMsg.config();
        eventBroker_.publishEvent<event::NewConfigReceived>(config);
        break;
      }
    default:
      std::cout << "Unknown request type" << std::endl;
      break;
  }
}

void UserInterface::handleSelfSpawned() {
  const auto &currentLevelData = gameData_.levelData().getLevel(worldState_->selfState().getCurrentLevel());
  const auto &regionName = gameData_.textZoneNameData().getRegionName(worldState_->selfState().position().regionId());

  broadcastCharacterSpawn();
  broadcastCharacterLevelUpdate(worldState_->selfState().getCurrentLevel(), currentLevelData.exp_C);
  broadcastCharacterExperienceUpdate(worldState_->selfState().getCurrentExperience(), worldState_->selfState().getCurrentSpExperience());
  broadcastCharacterSpUpdate(worldState_->selfState().getSkillPoints());
  broadcastCharacterNameUpdate(worldState_->selfState().name);
  broadcastGoldAmountUpdate(worldState_->selfState().getGold(), proto::broadcast::ItemLocation::kCharacterInventory);
  broadcastMovementEndedUpdate(worldState_->selfState().position());
  broadcastRegionNameUpdate(regionName);

  // Send entire inventory
  for (uint8_t inventorySlotIndex=0; inventorySlotIndex<worldState_->selfState().inventory.size(); ++inventorySlotIndex) {
    if (worldState_->selfState().inventory.hasItem(inventorySlotIndex)) {
      broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kCharacterInventory, worldState_->selfState().inventory, inventorySlotIndex);
    }
  }

  // Send avatar inventory too
  for (uint8_t inventorySlotIndex=0; inventorySlotIndex<worldState_->selfState().avatarInventory.size(); ++inventorySlotIndex) {
    if (worldState_->selfState().avatarInventory.hasItem(inventorySlotIndex)) {
      broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kAvatarInventory, worldState_->selfState().avatarInventory, inventorySlotIndex);
    }
  }
}

void UserInterface::handleCosSpawned(const event::CosSpawned &event) {
  // Send COS inventory
  auto it = worldState_->selfState().cosInventoryMap.find(event.cosGlobalId);
  if (it == worldState_->selfState().cosInventoryMap.end()) {
    // Received COS Spawned event, but dont have this COS
    return;
  }
  const auto &cosInventory = it->second;
  for (uint8_t inventorySlotIndex=0; inventorySlotIndex<cosInventory.size(); ++inventorySlotIndex) {
    if (cosInventory.hasItem(inventorySlotIndex)) {
      broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kCosInventory, cosInventory, inventorySlotIndex);
    }
  }
}

void UserInterface::handleEntitySpawned(const event::EntitySpawned &event) {
  const bool trackingEntity = worldState_->entityTracker().trackingEntity(event.globalId);
  if (!trackingEntity) {
    // Received entity spawned event, but we're not tracking this entity
    return;
  }
  const auto &entity = worldState_->getEntity<entity::Entity>(event.globalId);
  broadcastEntitySpawned(&entity);
  if (const auto *characterEntity = dynamic_cast<const entity::Character*>(&entity)) {
    if (characterEntity->lifeState == sro::entity::LifeState::kDead) {
      // Entity spawned in as dead
      // TODO: Create a more comprehensive entity proto which contains life state
      broadcastEntityLifeStateChanged(characterEntity->globalId, characterEntity->lifeState);
    }
  }

}

void UserInterface::handleEntityMovementBegan(const event::EntityMovementBegan &event) {
  const entity::MobileEntity &mobileEntity = worldState_->getEntity<entity::MobileEntity>(event.globalId);
  if (!mobileEntity.moving()) {
    // Got an entity movement began event, but it is not moving
    return;
  }
  const auto currentPosition = mobileEntity.position();
  if (mobileEntity.destinationPosition) {
    if (event.globalId == worldState_->selfState().globalId) {
      // TODO: We ought to combine these two UI functions
      broadcastMovementBeganUpdate(currentPosition, *worldState_->selfState().destinationPosition, worldState_->selfState().currentSpeed());
    } else {
      broadcastEntityMovementBegan(event.globalId, currentPosition, *mobileEntity.destinationPosition, mobileEntity.currentSpeed());
    }
  } else {
    if (event.globalId == worldState_->selfState().globalId) {
      // TODO: We ought to combine these two UI functions
      broadcastMovementBeganUpdate(currentPosition, worldState_->selfState().angle(), worldState_->selfState().currentSpeed());
    } else {
      broadcastEntityMovementBegan(event.globalId, currentPosition, mobileEntity.angle(), mobileEntity.currentSpeed());
    }
  }
}

void UserInterface::handleEntityMovementEnded(const event::EntityMovementEnded &event) {
  const entity::MobileEntity &mobileEntity = worldState_->getEntity<entity::MobileEntity>(event.globalId);
  if (event.globalId == worldState_->selfState().globalId) {
    // TODO: We ought to combine these two UI functions
    broadcastMovementEndedUpdate(mobileEntity.position());
  } else {
    broadcastEntityMovementEnded(event.globalId, mobileEntity.position());
  }
}

void UserInterface::handleEntityPositionUpdated(sro::scalar_types::EntityGlobalId globalId) {
  const entity::MobileEntity &mobileEntity = worldState_->getEntity<entity::MobileEntity>(globalId);
  if (mobileEntity.moving()) {
    // Should never happen while moving
    return;
  }

  // Not moving
  const auto currentPosition = mobileEntity.position();
  if (globalId == worldState_->selfState().globalId) {
    broadcastPositionChangedUpdate(currentPosition);
  } else {
    broadcastEntityPositionChanged(mobileEntity.globalId, currentPosition);
  }
}

void UserInterface::handleEntityNotMovingAngleChanged(sro::scalar_types::EntityGlobalId globalId) {
  if (globalId == worldState_->selfState().globalId) {
    // We only send the angle of the controlled character to the UI
    const entity::MobileEntity &mobileEntity = worldState_->getEntity<entity::MobileEntity>(globalId);
    broadcastNotMovingAngleChangedUpdate(mobileEntity.angle());
  }
}

void UserInterface::handleStorageInitialized() {
  for (sro::scalar_types::StorageIndexType storageSlotIndex=0; storageSlotIndex<worldState_->selfState().storage.size(); ++storageSlotIndex) {
    if (worldState_->selfState().storage.hasItem(storageSlotIndex)) {
      broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kStorage, worldState_->selfState().storage, storageSlotIndex);
    }
  }
}

void UserInterface::handleGuildStorageInitialized() {
  for (sro::scalar_types::StorageIndexType storageSlotIndex=0; storageSlotIndex<worldState_->selfState().guildStorage.size(); ++storageSlotIndex) {
    if (worldState_->selfState().guildStorage.hasItem(storageSlotIndex)) {
      broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kGuildStorage, worldState_->selfState().guildStorage, storageSlotIndex);
    }
  }
}

void UserInterface::handleInventoryUpdated(const event::InventoryUpdated &inventoryUpdatedEvent) {
  if (inventoryUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kCharacterInventory, worldState_->selfState().inventory, *inventoryUpdatedEvent.srcSlotNum);
  }
  if (inventoryUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kCharacterInventory, worldState_->selfState().inventory, *inventoryUpdatedEvent.destSlotNum);
  }
}

void UserInterface::handleAvatarInventoryUpdated(const event::AvatarInventoryUpdated &avatarInventoryUpdatedEvent) {
  if (avatarInventoryUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kAvatarInventory, worldState_->selfState().avatarInventory, *avatarInventoryUpdatedEvent.srcSlotNum);
  }
  if (avatarInventoryUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kAvatarInventory, worldState_->selfState().avatarInventory, *avatarInventoryUpdatedEvent.destSlotNum);
  }
}

void UserInterface::handleCosInventoryUpdated(const event::CosInventoryUpdated &cosInventoryUpdatedEvent) {
  auto it = worldState_->selfState().cosInventoryMap.find(cosInventoryUpdatedEvent.globalId);
  if (it == worldState_->selfState().cosInventoryMap.end()) {
    // COS inventory updated, but not tracking this COS
    return;
  }
  const auto &cosInventory = it->second;
  if (cosInventoryUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kCosInventory, cosInventory, *cosInventoryUpdatedEvent.srcSlotNum);
  }
  if (cosInventoryUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kCosInventory, cosInventory, *cosInventoryUpdatedEvent.destSlotNum);
  }
}

void UserInterface::handleStorageUpdated(const event::StorageUpdated &storageUpdatedEvent) {
  if (storageUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kStorage, worldState_->selfState().storage, *storageUpdatedEvent.srcSlotNum);
  }
  if (storageUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kStorage, worldState_->selfState().storage, *storageUpdatedEvent.destSlotNum);
  }
}

void UserInterface::handleGuildStorageUpdated(const event::GuildStorageUpdated &guildStorageUpdatedEvent) {
  if (guildStorageUpdatedEvent.srcSlotNum) {
    broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kGuildStorage, worldState_->selfState().guildStorage, *guildStorageUpdatedEvent.srcSlotNum);
  }
  if (guildStorageUpdatedEvent.destSlotNum) {
    broadcastItemUpdateForSlot(proto::broadcast::ItemLocation::kGuildStorage, worldState_->selfState().guildStorage, *guildStorageUpdatedEvent.destSlotNum);
  }
}

void UserInterface::handleWalkingPathUpdated(const event::WalkingPathUpdated &walkingPathUpdatedEvent) {
  std::vector<sro::Position> waypointSroPositions;
  waypointSroPositions.reserve(walkingPathUpdatedEvent.waypoints.size());
  for (const auto &waypoint : walkingPathUpdatedEvent.waypoints) {
    waypointSroPositions.emplace_back(waypoint.asSroPosition());
  }
  broadcastWalkingPathUpdated(waypointSroPositions);
}

void UserInterface::broadcastItemUpdateForSlot(proto::broadcast::ItemLocation itemLocation, const storage::Storage &itemStorage, const uint8_t slotIndex) {
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
  broadcastItemUpdate(itemLocation, slotIndex, quantity, itemName);
}

void UserInterface::broadcastCharacterSpawn() {
  proto::broadcast::BroadcastMessage broadcastMessage;
  broadcastMessage.mutable_characterspawn();
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterHpUpdate(uint32_t currentHp) {
  proto::broadcast::CharacterHpUpdate characterHpUpdate;
  characterHpUpdate.set_currenthp(currentHp);
  proto::broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_characterhpupdate() = characterHpUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterMpUpdate(uint32_t currentMp) {
  proto::broadcast::CharacterMpUpdate characterMpUpdate;
  characterMpUpdate.set_currentmp(currentMp);
  proto::broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_charactermpupdate() = characterMpUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterMaxHpMpUpdate(uint32_t maxHp, uint32_t maxMp) {
  proto::broadcast::CharacterMaxHpMpUpdate characterMaxHpMpUpdate;
  characterMaxHpMpUpdate.set_maxhp(maxHp);
  characterMaxHpMpUpdate.set_maxmp(maxMp);
  proto::broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_charactermaxhpmpupdate() = characterMaxHpMpUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterLevelUpdate(uint8_t currentLevel, int64_t expRequired) {
  proto::broadcast::CharacterLevelUpdate characterLevelUpdate;
  characterLevelUpdate.set_level(currentLevel);
  characterLevelUpdate.set_exprequired(expRequired);
  proto::broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_characterlevelupdate() = characterLevelUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience) {
  proto::broadcast::CharacterExperienceUpdate characterExperienceUpdate;
  characterExperienceUpdate.set_currentexperience(currentExperience);
  characterExperienceUpdate.set_currentspexperience(currentSpExperience);
  proto::broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_characterexperienceupdate() = characterExperienceUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterSpUpdate(uint32_t skillPoints) {
  proto::broadcast::CharacterSpUpdate characterSpUpdate;
  characterSpUpdate.set_skillpoints(skillPoints);
  proto::broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_characterspupdate() = characterSpUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterNameUpdate(std::string_view characterName) {
  proto::broadcast::CharacterNameUpdate characterNameUpdate;
  characterNameUpdate.set_name(std::string(characterName));
  proto::broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_characternameupdate() = characterNameUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastGoldAmountUpdate(uint64_t goldAmount, proto::broadcast::ItemLocation goldLocation) {
  proto::broadcast::GoldAmountUpdate goldAmountUpdate;
  goldAmountUpdate.set_goldamount(goldAmount);
  goldAmountUpdate.set_goldlocation(goldLocation);
  proto::broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_goldamountupdate() = goldAmountUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastPositionChangedUpdate(const sro::Position &currentPosition) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  auto *msg = broadcastMessage.mutable_characterpositionchanged();
  setPosition(msg->mutable_position(), currentPosition);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastMovementBeganUpdate(const sro::Position &srcPosition, const sro::Position &destPosition, float speed) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  setCharacterMovementBegan(broadcastMessage.mutable_charactermovementbegan(), srcPosition, destPosition, speed);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastMovementBeganUpdate(const sro::Position &srcPosition, sro::Angle angle, float speed) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  setCharacterMovementBegan(broadcastMessage.mutable_charactermovementbegan(), srcPosition, angle, speed);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastMovementEndedUpdate(const sro::Position &currentPosition) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  setCharacterMovementEnded(broadcastMessage.mutable_charactermovementended(), currentPosition);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastNotMovingAngleChangedUpdate(sro::Angle angle) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  broadcastMessage.mutable_characternotmovinganglechanged()->set_angle(angle);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastRegionNameUpdate(std::string_view regionName) {
  proto::broadcast::RegionNameUpdate regionNameUpdate;
  regionNameUpdate.set_name(std::string(regionName));
  proto::broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_regionnameupdate() = regionNameUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastItemUpdate(proto::broadcast::ItemLocation itemLocation, uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  auto *itemUpdate = broadcastMessage.mutable_itemupdate();
  itemUpdate->set_itemlocation(itemLocation);
  itemUpdate->set_slotindex(slotIndex);
  itemUpdate->set_quantity(quantity);
  if (quantity != 0) {
    if (!itemName) {
      throw std::runtime_error("Quantity of item positive, but we were given no item name");
    }
    itemUpdate->set_itemname(*itemName);
  }
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntitySpawned(const entity::Entity *entity) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  auto *entitySpawnedMsg = broadcastMessage.mutable_entityspawned();
  entitySpawnedMsg->set_globalid(entity->globalId);
  setPosition(entitySpawnedMsg->mutable_position(), entity->position());
  setEntity(entitySpawnedMsg->mutable_entity(), entity);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityDespawned(uint32_t globalId) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  auto *entityDespawnedMsg = broadcastMessage.mutable_entitydespawned();
  entityDespawnedMsg->set_globalid(globalId);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityPositionChanged(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &position) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  auto *entityPositionChangedMsg = broadcastMessage.mutable_entitypositionchanged();
  entityPositionChangedMsg->set_globalid(globalId);
  setPosition(entityPositionChangedMsg->mutable_position(), position);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityMovementBegan(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &srcPosition, const sro::Position &destPosition, float speed) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  proto::broadcast::EntityMovementBegan *entityMovementBegan = broadcastMessage.mutable_entitymovementbegan();
  entityMovementBegan->set_globalid(globalId);
  setCharacterMovementBegan(entityMovementBegan->mutable_charactermovementbegan(), srcPosition, destPosition, speed);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityMovementBegan(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &srcPosition, sro::Angle angle, float speed) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  proto::broadcast::EntityMovementBegan *entityMovementBegan = broadcastMessage.mutable_entitymovementbegan();
  entityMovementBegan->set_globalid(globalId);
  setCharacterMovementBegan(entityMovementBegan->mutable_charactermovementbegan(), srcPosition, angle, speed);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityMovementEnded(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &currentPosition) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  proto::broadcast::EntityMovementEnded *entityMovementEnded = broadcastMessage.mutable_entitymovementended();
  entityMovementEnded->set_globalid(globalId);
  setCharacterMovementEnded(entityMovementEnded->mutable_charactermovementended(), currentPosition);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityLifeStateChanged(const sro::scalar_types::EntityGlobalId globalId, const sro::entity::LifeState lifeState) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  proto::broadcast::EntityLifeStateChanged *entityLifeStateChanged = broadcastMessage.mutable_entitylifestatechanged();
  entityLifeStateChanged->set_globalid(globalId);
  entityLifeStateChanged->set_lifestate(lifeStateToProto(lifeState));
  broadcast(broadcastMessage);
}

void UserInterface::broadcastTrainingAreaSet(const entity::Geometry *trainingAreaGeometry) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  proto::broadcast::TrainingAreaSet *trainingAreaSet = broadcastMessage.mutable_trainingareaset();
  if (const auto *trainingAreaCircle = dynamic_cast<const entity::Circle*>(trainingAreaGeometry)) {
    proto::broadcast::Circle *circle = trainingAreaSet->mutable_circle();
    setPosition(circle->mutable_center(), trainingAreaCircle->center());
    circle->set_radius(trainingAreaCircle->radius());
  } else {
    throw std::runtime_error("Unsupported training area geometry");
  }
  broadcast(broadcastMessage);
}

void UserInterface::broadcastTrainingAreaReset() {
  proto::broadcast::BroadcastMessage broadcastMessage;
  broadcastMessage.mutable_trainingareareset();
  broadcast(broadcastMessage);
}

void UserInterface::broadcastStateMachineCreated(const std::string &stateMachineName) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  proto::broadcast::StateMachineCreated *stateMachineCreated = broadcastMessage.mutable_statemachinecreated();
  stateMachineCreated->set_name(stateMachineName);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastStateMachineDestroyed() {
  proto::broadcast::BroadcastMessage broadcastMessage;
  broadcastMessage.mutable_statemachinedestroyed();
  broadcast(broadcastMessage);
}

void UserInterface::broadcastWalkingPathUpdated(const std::vector<sro::Position> &waypoints) {
  proto::broadcast::BroadcastMessage broadcastMessage;
  proto::broadcast::WalkingPathUpdated *walkingPathUpdated = broadcastMessage.mutable_walkingpathupdated();
  for (const auto &waypoint : waypoints) {
    setPosition(walkingPathUpdated->add_waypoints(), waypoint);
  }
  broadcast(broadcastMessage);
}

void UserInterface::broadcast(const proto::broadcast::BroadcastMessage &broadcastProto) {
  zmq::message_t message;
  message.rebuild(broadcastProto.ByteSizeLong());
  broadcastProto.SerializeToArray(message.data(), message.size());
  const auto res = publisher_.send(message, zmq::send_flags::none);
  // TODO: Check result

  // Old method:
  // zmq::message_t msg;
  // std::string str;
  // broadcastMessage.SerializeToString(&str);
  // msg.rebuild(str.data(), str.size());
  // auto res = publisher_.send(msg, zmq::send_flags::none);
}

void UserInterface::setPosition(proto::broadcast::Position *msg, const sro::Position &pos) const {
  msg->set_regionid(pos.regionId());
  msg->set_x(pos.xOffset());
  msg->set_y(pos.yOffset());
  msg->set_z(pos.zOffset());
}

void UserInterface::setCharacterMovementBegan(proto::broadcast::CharacterMovementBegan *msg, const sro::Position &srcPosition, const sro::Position &destPosition, const float speed) const {
  setPosition(msg->mutable_currentposition(), srcPosition);
  setPosition(msg->mutable_destinationposition(), destPosition);
  msg->set_speed(speed);
}

void UserInterface::setCharacterMovementBegan(proto::broadcast::CharacterMovementBegan *msg, const sro::Position &srcPosition, const sro::Angle angle, const float speed) const {
  setPosition(msg->mutable_currentposition(), srcPosition);
  msg->set_destinationangle(angle);
  msg->set_speed(speed);
}

void UserInterface::setCharacterMovementEnded(proto::broadcast::CharacterMovementEnded *msg, const sro::Position &currentPosition) const {
  setPosition(msg->mutable_currentposition(), currentPosition);
}

void UserInterface::setEntity(proto::entity::Entity *msg, const entity::Entity *entity) const {
  msg->set_reference_id(entity->refObjId);
  const auto entityType = entity->entityType();
  if (entityType == entity::EntityType::kSelf) {
    msg->mutable_self();
  } else if (entityType == entity::EntityType::kCharacter) {
    msg->mutable_character();
  } else if (entityType == entity::EntityType::kPlayerCharacter) {
    msg->mutable_player_character();
  } else if (entityType == entity::EntityType::kNonplayerCharacter) {
    msg->mutable_nonplayer_character();
  } else if (entityType == entity::EntityType::kPortal) {
    msg->mutable_portal();
  } else if (entityType == entity::EntityType::kMonster) {
    auto *monsterMsg = msg->mutable_monster();
    const auto *entityAsMonster = dynamic_cast<const entity::Monster*>(entity);
    if (entityAsMonster == nullptr) {
      throw std::runtime_error("Entity type is Monster, but dynamic cast to monster failed");
    }
    switch (entityAsMonster->rarity) {
      case sro::entity::MonsterRarity::kGeneral:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kGeneral);
        break;
      case sro::entity::MonsterRarity::kChampion:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kChampion);
        break;
      case sro::entity::MonsterRarity::kUnique:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kUnique);
        break;
      case sro::entity::MonsterRarity::kGiant:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kGiant);
        break;
      case sro::entity::MonsterRarity::kTitan:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kTitan);
        break;
      case sro::entity::MonsterRarity::kElite:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kElite);
        break;
      case sro::entity::MonsterRarity::kEliteStrong:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kEliteStrong);
        break;
      case sro::entity::MonsterRarity::kUnique2:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kUnique2);
        break;
      case sro::entity::MonsterRarity::kGeneralParty:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kGeneralParty);
        break;
      case sro::entity::MonsterRarity::kChampionParty:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kChampionParty);
        break;
      case sro::entity::MonsterRarity::kUniqueParty:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kUniqueParty);
        break;
      case sro::entity::MonsterRarity::kGiantParty:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kGiantParty);
        break;
      case sro::entity::MonsterRarity::kTitanParty:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kTitanParty);
        break;
      case sro::entity::MonsterRarity::kEliteParty:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kEliteParty);
        break;
      case sro::entity::MonsterRarity::kEliteStrongParty:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kEliteStrongParty);
        break;
      case sro::entity::MonsterRarity::kUnique2Party:
        monsterMsg->set_rarity(proto::entity::MonsterRarity::kUnique2Party);
        break;
      default:
        throw std::runtime_error("Unknown monster rarity");
    }
  } else if (entityType == entity::EntityType::kItem) {
    auto *itemMsg = msg->mutable_item();
    const auto *entityAsItem = dynamic_cast<const entity::Item*>(entity);
    if (entityAsItem == nullptr) {
      throw std::runtime_error("Entity type is Item, but dynamic cast to item failed");
    }
    switch (entityAsItem->rarity) {
      case sro::entity::ItemRarity::kWhite:
        itemMsg->set_rarity(proto::entity::ItemRarity::kWhite);
        break;
      case sro::entity::ItemRarity::kBlue:
        itemMsg->set_rarity(proto::entity::ItemRarity::kBlue);
        break;
      case sro::entity::ItemRarity::kSox:
        itemMsg->set_rarity(proto::entity::ItemRarity::kSox);
        break;
      case sro::entity::ItemRarity::kSet:
        itemMsg->set_rarity(proto::entity::ItemRarity::kSet);
        break;
      case sro::entity::ItemRarity::kRareSet:
        itemMsg->set_rarity(proto::entity::ItemRarity::kRareSet);
        break;
      case sro::entity::ItemRarity::kLegend:
        itemMsg->set_rarity(proto::entity::ItemRarity::kLegend);
        break;
      default:
        throw std::runtime_error("Unknown item rarity");
    }
  }
}

} // namespace ui