#include "characterInfoModule.hpp"
#include "../packet/opcode.hpp"
#include "../packet/parsing/clientAgentChatRequest.hpp"
#include "../packet/building/clientAgentInventoryItemUseRequest.hpp"
#include "../packet/building/clientAgentInventoryOperationRequest.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <regex>

// #define ENFORCE_PURIFICATION_PILL_COOLDOWN
namespace module {

void printItem(uint8_t slot, const storage::Item *item, const pk2::GameData &gameData);
int bitNum(packet::enums::AbnormalStateFlag flag);
std::string toStr(packet::enums::AbnormalStateFlag state);
uint16_t makeTypeId(const uint16_t typeId1, const uint16_t typeId2, const uint16_t typeId3, const uint16_t typeId4);

//======================================================================================================
//======================================================================================================
//======================================================================================================

CharacterInfoModule::CharacterInfoModule(state::Entity &entityState,
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
  auto packetHandleFunction = std::bind(&CharacterInfoModule::handlePacket, this, std::placeholders::_1);
  // Client packets
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentInventoryOperationRequest, packetHandleFunction);
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentChatRequest, packetHandleFunction);
  // Server packets
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentCharacterData, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_HPMP_UPDATE, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_STATS, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_ITEM_USE, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_AGENT_ABNORMAL_INFO, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_ITEM_MOVEMENT, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_AGENT_ENTITY_GROUPSPAWN_DATA, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_SPAWN, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_DESPAWN, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateState, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateMoveSpeed, packetHandleFunction);

  auto eventHandleFunction = std::bind(&CharacterInfoModule::handleEvent, this, std::placeholders::_1);
  // TODO: Save subscription ID to possibly unsubscribe in the future
  eventBroker_.subscribeToEvent(event::EventCode::kHpPotionCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMpPotionCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kVigorPotionCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kUniversalPillCooldownEnded, eventHandleFunction);
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
  eventBroker_.subscribeToEvent(event::EventCode::kPurificationPillCooldownEnded, eventHandleFunction);
#endif
  eventBroker_.subscribeToEvent(event::EventCode::kHpPercentChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMpPercentChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStatesChanged, eventHandleFunction);

  eventBroker_.subscribeToEvent(event::EventCode::kDropGold, eventHandleFunction);
}

void CharacterInfoModule::handleEvent(const event::Event *event) {
  std::unique_lock<std::mutex> contentionProtectionLock(contentionProtectionMutex_);
  const auto eventCode = event->eventCode;
  switch (eventCode) {
    case event::EventCode::kHpPotionCooldownEnded:
    case event::EventCode::kMpPotionCooldownEnded:
    case event::EventCode::kVigorPotionCooldownEnded:
      handlePotionCooldownEnded(eventCode);
      break;
    case event::EventCode::kUniversalPillCooldownEnded:
    case event::EventCode::kPurificationPillCooldownEnded:
      handlePillCooldownEnded(eventCode);
      break;
    case event::EventCode::kHpPercentChanged:
    case event::EventCode::kMpPercentChanged:
      handleVitalsChanged();
      break;
    case event::EventCode::kStatesChanged:
      handleStatesChanged();
      break;
    case event::EventCode::kDropGold:
      handleDropGold(event);
      break;
    default:
      std::cout << "Unhandled event subscribed to. Code:" << static_cast<int>(eventCode) << '\n';
      break;
  }
}

void CharacterInfoModule::handleDropGold(const event::Event *event) {
  const event::DropGold &dropGoldEvent = dynamic_cast<const event::DropGold&>(*event);
  std::cout << "Asked to drop " << dropGoldEvent.goldAmount << " gold" << std::endl;
  goldDropAmount_ = dropGoldEvent.goldAmount;
  goldDropRemaining_ = dropGoldEvent.goldDropCount;
  broker_.injectPacket(packet::building::ClientAgentInventoryOperationRequest::packet(dropGoldEvent.goldAmount), PacketContainer::Direction::kClientToServer);
}

void CharacterInfoModule::handlePillCooldownEnded(const event::EventCode eventCode) {
  if (eventCode == event::EventCode::kUniversalPillCooldownEnded) {
    selfState_.resetUniversalPillEventId();
    checkIfNeedToUsePill();
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
  } else if (eventCode == event::EventCode::kPurificationPillCooldownEnded) {
    std::cout << "kPurificationPillCooldownEnded " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() << "\n";
    selfState_.resetPurificationPillEventId();
    checkIfNeedToUsePill();
#endif
  }
}

void CharacterInfoModule::handlePotionCooldownEnded(const event::EventCode eventCode) {
  if (eventCode == event::EventCode::kHpPotionCooldownEnded) {
    selfState_.resetHpPotionEventId();
    checkIfNeedToHeal();
  } else if (eventCode == event::EventCode::kMpPotionCooldownEnded) {
    selfState_.resetMpPotionEventId();
    checkIfNeedToHeal();
  } else if (eventCode == event::EventCode::kVigorPotionCooldownEnded) {
    selfState_.resetVigorPotionEventId();
    checkIfNeedToHeal();
  }
}

void CharacterInfoModule::handleVitalsChanged() {
  checkIfNeedToHeal();

  // Broadcast message to UI
  if (!selfState_.maxHp() || !selfState_.maxMp()) {
    // Dont yet know our max
    std::cout << "handleVitalsChanged: dont know max hp or mp\n";
    return;
  }

  broadcast::HpMpUpdate hpMpUpdate;
  hpMpUpdate.set_currenthp(selfState_.hp());
  hpMpUpdate.set_maxhp(*selfState_.maxHp());
  hpMpUpdate.set_currentmp(selfState_.mp());
  hpMpUpdate.set_maxmp(*selfState_.maxMp());
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_hpmpupdate() = hpMpUpdate;
  userInterface_.broadcast(broadcastMessage);
}

void CharacterInfoModule::handleStatesChanged() {
  checkIfNeedToUsePill();
}

bool CharacterInfoModule::handlePacket(const PacketContainer &packet) {
  std::unique_ptr<packet::parsing::ParsedPacket> parsedPacket;
  try {
    parsedPacket = packetParser_.parsePacket(packet);
  } catch (std::exception &ex) {
    std::cerr << "[CharacterInfoModule] Failed to parse packet " << std::hex << packet.opcode << std::dec << "\n  Error: \"" << ex.what() << "\"\n";
    return true;
  }

  if (!parsedPacket) {
    // Not yet parsing this packet
    return true;
  }

  //======================================================================================================
  //============================================Client Packets============================================
  //======================================================================================================

  auto *clientItemMove = dynamic_cast<packet::parsing::ParsedClientItemMove*>(parsedPacket.get());
  if (clientItemMove != nullptr) {
    clientItemMoveReceived(*clientItemMove);
    return true;
  }

  auto *clientChat = dynamic_cast<packet::parsing::ClientAgentChatRequest*>(parsedPacket.get());
  if (clientChat != nullptr) {
    return true;
  }

  //======================================================================================================
  //============================================Server Packets============================================
  //======================================================================================================

  auto *charData = dynamic_cast<packet::parsing::ParsedServerAgentCharacterData*>(parsedPacket.get());
  if (charData != nullptr) {
    serverAgentCharacterDataReceived(*charData);
    return true;
  }

  auto *hpMpUpdate = dynamic_cast<packet::parsing::ParsedServerHpMpUpdate*>(parsedPacket.get());
  if (hpMpUpdate != nullptr) {
    entityUpdateReceived(*hpMpUpdate);
    return true;
  }

  auto *serverUseItemUpdate = dynamic_cast<packet::parsing::ParsedServerUseItem*>(parsedPacket.get());
  if (serverUseItemUpdate != nullptr) {
    serverUseItemReceived(*serverUseItemUpdate);
    return true;
  }

  auto *statUpdate = dynamic_cast<packet::parsing::ParsedServerAgentCharacterUpdateStats*>(parsedPacket.get());
  if (statUpdate != nullptr) {
    statUpdateReceived(*statUpdate);
    return true;
  }

  auto *abnormalInfo = dynamic_cast<packet::parsing::ParsedServerAbnormalInfo*>(parsedPacket.get());
  if (abnormalInfo != nullptr) {
    abnormalInfoReceived(*abnormalInfo);
    return true;
  }

  auto *serverItemMove = dynamic_cast<packet::parsing::ParsedServerItemMove*>(parsedPacket.get());
  if (serverItemMove != nullptr) {
    serverItemMoveReceived(*serverItemMove);
    return true;
  }

  auto *groupSpawn = dynamic_cast<packet::parsing::ParsedServerAgentGroupSpawn*>(parsedPacket.get());
  if (groupSpawn != nullptr) {
    serverAgentGroupSpawnReceived(*groupSpawn);
    return true;
  }

  auto *spawn = dynamic_cast<packet::parsing::ParsedServerAgentSpawn*>(parsedPacket.get());
  if (spawn != nullptr) {
    serverAgentSpawnReceived(*spawn);
    return true;
  }

  auto *despawn = dynamic_cast<packet::parsing::ParsedServerAgentDespawn*>(parsedPacket.get());
  if (despawn != nullptr) {
    serverAgentDespawnReceived(*despawn);
    return true;
  }

  auto *entityUpdateState = dynamic_cast<packet::parsing::ServerAgentEntityUpdateState*>(parsedPacket.get());
  if (entityUpdateState != nullptr) {
    serverAgentEntityUpdateStateReceived(*entityUpdateState);
    return true;
  }

  auto *entityUpdateMoveSpeed = dynamic_cast<packet::parsing::ServerAgentEntityUpdateMoveSpeed*>(parsedPacket.get());
  if (entityUpdateMoveSpeed != nullptr) {
    serverAgentEntityUpdateMoveSpeedReceived(*entityUpdateMoveSpeed);
    return true;
  }
  
  //======================================================================================================

  std::cout << "CharacterInfoModule: Unhandled packet subscribed to\n";
  return true;
}

void CharacterInfoModule::serverAgentEntityUpdateMoveSpeedReceived(const packet::parsing::ServerAgentEntityUpdateMoveSpeed &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  if (packet.globalId() == selfState_.globalId()) {
    // Our speed was updated
    std::cout << "Our speed was updated from " << selfState_.walkSpeed() << ',' << selfState_.runSpeed() << " to " << packet.walkSpeed() << ',' << packet.runSpeed() << '\n';
    selfState_.setSpeed(packet.walkSpeed(), packet.runSpeed());
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kCharacterSpeedUpdated));
  }
}

void CharacterInfoModule::serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  if (selfState_.spawned() && packet.gId() == selfState_.globalId()) {
    if (packet.stateType() == packet::parsing::StateType::kBodyState) {
      selfState_.setBodyState(static_cast<packet::enums::BodyState>(packet.state()));
    } else if (packet.stateType() == packet::parsing::StateType::kLifeState) {
      selfState_.setLifeState(static_cast<packet::enums::LifeState>(packet.state()));
      if (static_cast<packet::enums::LifeState>(packet.state()) == packet::enums::LifeState::kDead) {
        std::cout << "CharacterInfoModule: We died, clearing used item queue" << std::endl;
        selfState_.clearUsedItemQueue();
      }
    } else if (packet.stateType() == packet::parsing::StateType::kMotionState) {
      if (static_cast<packet::enums::MotionState>(packet.state()) == packet::enums::MotionState::kWalk) {
        std::cout << "Motion state update to walk\n";
      } else if (static_cast<packet::enums::MotionState>(packet.state()) == packet::enums::MotionState::kRun) {
        std::cout << "Motion state update to run\n";
      } else if (static_cast<packet::enums::MotionState>(packet.state()) == packet::enums::MotionState::kSit) {
        std::cout << "Motion state update to sit\n";
      } else {
        std::cout << "Motion state update to " << static_cast<int>(packet.state()) << '\n';
      }
      selfState_.setMotionState(static_cast<packet::enums::MotionState>(packet.state()));
    }
  }
}

void CharacterInfoModule::trackObject(std::shared_ptr<packet::parsing::Object> obj) {
  entityState_.trackEntity(obj);
  // printf("[+++] (%5d)  ", entityState_.size());
  // packet::parsing::printObj(obj.get(), gameData_);
}

void CharacterInfoModule::stopTrackingObject(uint32_t gId) {
  if (entityState_.trackingEntity(gId)) {
    auto objPtr = entityState_.getEntity(gId);
    // printf("[---] (%5d)  ", entityState_.size()-1);
    // packet::parsing::printObj(objPtr, gameData_);
    entityState_.stopTrackingEntity(gId);
  } else {
    std::cout << "Asked to despawn something that we werent tracking\n";
  }
}

void CharacterInfoModule::serverAgentSpawnReceived(packet::parsing::ParsedServerAgentSpawn &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  if (packet.object()) {
    trackObject(packet.object());
  } else {
    std::cout << "Object spawned which we cannot track\n";
  }
}

void CharacterInfoModule::serverAgentDespawnReceived(packet::parsing::ParsedServerAgentDespawn &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  stopTrackingObject(packet.gId());
}

void CharacterInfoModule::clientItemMoveReceived(const packet::parsing::ParsedClientItemMove &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  const auto itemMovement = packet.movement();
  if (itemMovement.type == packet::enums::ItemMovementType::kBuyFromNPC) {
    // User is buying something from the store
    selfState_.setUserPurchaseRequest(itemMovement);
  }
}

void CharacterInfoModule::serverAgentGroupSpawnReceived(const packet::parsing::ParsedServerAgentGroupSpawn &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  if (packet.groupSpawnType() == packet::parsing::GroupSpawnType::kSpawn) {
    for (auto obj : packet.objects()) {
      trackObject(obj);
    }
  } else {
    for (auto gId : packet.despawns()) {
      stopTrackingObject(gId);
    }
  }
}

std::shared_ptr<storage::Item> createItemFromScrap(const pk2::ref::ScrapOfPackageItem &itemScrap, const pk2::ref::Item &itemRef) {
  std::shared_ptr<storage::Item> item(storage::newItemByTypeData(itemRef));

  storage::ItemExpendable *itemExpendable;
  storage::ItemEquipment *itemEquipment;
  storage::ItemStone *itemStone;
  storage::ItemCosGrowthSummoner *itemCosGrowthSummoner;
  storage::ItemStorage *itemStorage;
  storage::ItemMonsterCapsule *itemMonsterCapsule;
  storage::ItemCosAbilitySummoner *itemCosAbilitySummoner;
  storage::ItemMagicPop *itemMagicPop;
  if ((itemEquipment = dynamic_cast<storage::ItemEquipment*>(item.get())) != nullptr) {
    itemEquipment->optLevel = itemScrap.optLevel;
    itemEquipment->variance = itemScrap.variance;
    itemEquipment->durability = itemScrap.data;
    for (int i=0; i<itemScrap.magParamNum; ++i) {
      storage::ItemMagicParam param;
      auto paramData = itemScrap.magParams[i];
      uint64_t &data = reinterpret_cast<uint64_t&>(paramData);
      param.type = data & 0xFFFF;
      param.value = (data >> 32) & 0xFFFF;
      itemEquipment->magicParams.emplace_back(std::move(param));
    }
  }
  if ((itemExpendable = dynamic_cast<storage::ItemExpendable*>(item.get())) != nullptr) {
    itemExpendable->quantity = 0; // Arbitrary
  }
  if ((itemStone = dynamic_cast<storage::ItemStone*>(item.get())) != nullptr) {
    // TODO: Verify. No examples in current server
    itemStone->attributeAssimilationProbability = 0;
  }
  if ((itemCosGrowthSummoner = dynamic_cast<storage::ItemCosGrowthSummoner*>(item.get())) != nullptr) {
    // TODO: Verify. Pserver db data is all 0s for Wolf
    itemCosGrowthSummoner->lifeState = storage::CosLifeState::kInactive; // Based on buying then checking data
    itemCosGrowthSummoner->refObjID = 0; // Has no level, no item to ref to
  }
  if ((itemStorage = dynamic_cast<storage::ItemStorage*>(item.get())) != nullptr) {
    // TODO: Verify. No examples in current server
    itemStorage->quantity = 0;
  }
  if ((itemMonsterCapsule = dynamic_cast<storage::ItemMonsterCapsule*>(item.get())) != nullptr) {
    // TODO: Verify. No examples in current server
    itemMonsterCapsule->refObjID = 0; // Total guess
  }
  if ((itemCosAbilitySummoner = dynamic_cast<storage::ItemCosAbilitySummoner*>(item.get())) != nullptr) {
    // TODO: Verify. No examples in current server
    itemCosAbilitySummoner->secondsToRentEndTime = 0; // Based on buying from item mall and looking at data
  }
  if ((itemMagicPop = dynamic_cast<storage::ItemMagicPop*>(item.get())) != nullptr) {
    // TODO: Verify. No examples in current server
  }
  return item;
}

void CharacterInfoModule::printGold() {
  printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
  printf(" $$  Gold: %12llu  $$ \n", selfState_.getGold());
  printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
}

void CharacterInfoModule::serverItemMoveReceived(const packet::parsing::ParsedServerItemMove &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  const std::vector<packet::parsing::ItemMovement> &itemMovements = packet.itemMovements();
  for (const auto &movement : itemMovements) {
    if (movement.type == packet::enums::ItemMovementType::kWithinInventory) {
      selfState_.inventory.moveItem(movement.srcSlot, movement.destSlot, movement.quantity);
      //TODO: Add event in other places
      eventBroker_.publishEvent(std::make_unique<event::InventorySlotUpdated>(movement.srcSlot));
      eventBroker_.publishEvent(std::make_unique<event::InventorySlotUpdated>(movement.destSlot));
    } else if (movement.type == packet::enums::ItemMovementType::kWithinStorage) {
      // Not handling because we dont parse the storage init packet
      // moveItem(storage_, movement.srcSlot, movement.destSlot, movement.quantity);
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
            std::cout << "Bought " << movement.quantity << " x \"" << itemInfo.refItemCodeName << "\" from \"" << npcName << "\"\n";
            const auto &itemRef = gameData_.itemData().getItemByCodeName128(itemInfo.refItemCodeName);
            if (movement.destSlots.size() == 1) {
              // Just a single item or single stack
              auto item = createItemFromScrap(itemInfo, itemRef);
              storage::ItemExpendable *itemExp = dynamic_cast<storage::ItemExpendable*>(item.get());
              if (itemExp != nullptr) {
                itemExp->quantity = movement.quantity;
              }
              selfState_.inventory.addItem(movement.destSlots[0], item);
              printItem(movement.destSlots[0], item.get(), gameData_);
              std::cout << '\n';
            } else {
              // Multiple destination slots, must be unstackable items like equipment
              for (auto destSlot : movement.destSlots) {
                auto item = createItemFromScrap(itemInfo, itemRef);
                selfState_.inventory.addItem(destSlot, item);
                printItem(movement.destSlot, item.get(), gameData_);
                std::cout << '\n';
              }
            }
          }
        }
        selfState_.resetUserPurchaseRequest();
      } else {
        std::cout << "kBuyFromNPC but we dont have the data from the client packet\n";
        // TODO: Introduce unknown item concept?
      }
    } else if (movement.type == packet::enums::ItemMovementType::kSellToNPC) {
      if (selfState_.inventory.hasItem(movement.srcSlot)) {
        bool soldEntireStack = true;
        auto item = selfState_.inventory.getItem(movement.srcSlot);
        storage::ItemExpendable *itemExpendable;
        if ((itemExpendable = dynamic_cast<storage::ItemExpendable*>(item)) != nullptr) {
          if (itemExpendable->quantity != movement.quantity) {
            std::cout << "Sold only some of this item " << itemExpendable->quantity << " -> " << itemExpendable->quantity-movement.quantity << '\n';
            soldEntireStack = false;
            itemExpendable->quantity -= movement.quantity;
            auto clonedItem = storage::cloneItem(item);
            dynamic_cast<storage::ItemExpendable*>(clonedItem.get())->quantity = movement.quantity;
            selfState_.buybackQueue.addItem(clonedItem);
          }
        }
        if (soldEntireStack) {
          std::cout << "Sold entire \"stack\"\n";
          auto item = selfState_.inventory.withdrawItem(movement.srcSlot);
          selfState_.buybackQueue.addItem(item);
        }
      } else {
        std::cout << "Sold an item from a slot that we didnt have item data for\n";
      }
      std::cout << "Current buyback queue:\n";
      for (uint8_t slotNum=0; slotNum<selfState_.buybackQueue.size(); ++slotNum) {
        printItem(slotNum, selfState_.buybackQueue.getItem(slotNum), gameData_);
      }
      std::cout << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kBuyback) {
      if (selfState_.buybackQueue.hasItem(movement.srcSlot)) {
        if (!selfState_.inventory.hasItem(movement.destSlot)) {
          const auto itemPtr = selfState_.buybackQueue.getItem(movement.srcSlot);
          bool boughtBackAll = true;
          if (movement.quantity > 1) {
            storage::ItemExpendable *itemExpendable = dynamic_cast<storage::ItemExpendable*>(itemPtr);
            if (itemExpendable != nullptr) {
              if (itemExpendable->quantity > movement.quantity) {
                std::cout << "Only buying back a partial amount from the buyback slot. Didnt know this was possible (" << movement.quantity << '/' << itemExpendable->quantity << ")\n";
                boughtBackAll = false;
                auto clonedItem = storage::cloneItem(itemPtr);
                itemExpendable->quantity -= movement.quantity;
                dynamic_cast<storage::ItemExpendable*>(clonedItem.get())->quantity = movement.quantity;
                selfState_.inventory.addItem(movement.destSlot, clonedItem);
                std::cout << "Added item to inventory\n";
                printItem(movement.destSlot, clonedItem.get(), gameData_);
                std::cout << '\n';
              }
            }
          }
          if (boughtBackAll) {
            auto item = selfState_.buybackQueue.withdrawItem(movement.srcSlot);
            selfState_.inventory.addItem(movement.destSlot, item);
            std::cout << "Bought back entire stack\n";
            printItem(movement.destSlot, item.get(), gameData_);
          }
        } else {
          std::cout << "Bought back item is being moved into a slot that's already occupied\n";
        }
      } else {
        std::cout << "Bought back an item that we werent tracking\n";
      }
      std::cout << "Current buyback queue:\n";
      for (uint8_t slotNum=0; slotNum<selfState_.buybackQueue.size(); ++slotNum) {
        printItem(slotNum, selfState_.buybackQueue.getItem(slotNum), gameData_);
      }
      std::cout << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kPickItem) {
      if (movement.destSlot == packet::parsing::ItemMovement::kGoldSlot) {
        std::cout << "Picked " << movement.goldPickAmount << " gold\n";
        selfState_.addGold(movement.goldPickAmount);
        printGold();
        std::cout << '\n';
      } else {
        if (movement.pickedItem != nullptr) {
          std::cout << "Picked an item\n";
          if (selfState_.inventory.hasItem(movement.destSlot)) {
            std::cout << "Already something here\n";
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
              std::cout << "Item added to stack\n";
            } else {
              std::cout << "Error: Item couldnt be added to the stack\n";
            }
          } else {
            std::cout << "New item!\n";
            selfState_.inventory.addItem(movement.destSlot, movement.pickedItem);
            std::cout << "Item " << (selfState_.inventory.hasItem(movement.destSlot) ? "was " : "was not ") << "successfully added\n";
          }
          printItem(movement.destSlot, movement.pickedItem.get(), gameData_);
        } else {
          std::cout << "Error: Picked an item, but the pickedItem is a nullptr\n";
        }
      }
      // This would be a good time to try to use a pill, potion, return scroll, etc.
    } else if (movement.type == packet::enums::ItemMovementType::kDropItem) {
      std::cout << "Dropped an item\n";
      if (selfState_.inventory.hasItem(movement.srcSlot)) {
        std::cout << "Dropping ";
        auto itemPtr = selfState_.inventory.withdrawItem(movement.srcSlot);
        printItem(movement.srcSlot, itemPtr.get(), gameData_);
        std::cout << "Item " << (!selfState_.inventory.hasItem(movement.srcSlot) ? "was " : "was not ") << "successfully dropped\n";
      } else {
        std::cout << "Error: But there's no item in this inventory slot\n";
      }
    } else if (movement.type == packet::enums::ItemMovementType::kGoldDrop) {
      selfState_.subtractGold(movement.goldAmount);
      std::cout << "Dropped " << movement.goldAmount << " gold\n";
      printGold();
      std::cout << '\n';
      if (--goldDropRemaining_ > 0) {
        eventBroker_.publishEvent(std::make_unique<event::DropGold>(goldDropAmount_, goldDropRemaining_));
      }
    } else if (movement.type == packet::enums::ItemMovementType::kGoldStorageWithdraw) {
      selfState_.addGold(movement.goldAmount);
      std::cout << "Withdrew " << movement.goldAmount << " gold from storage\n";
      printGold();
      std::cout << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kGoldStorageDeposit) {
      selfState_.subtractGold(movement.goldAmount);
      std::cout << "Deposited " << movement.goldAmount << " gold into storage\n";
      printGold();
      std::cout << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kGoldGuildStorageDeposit) {
      selfState_.subtractGold(movement.goldAmount);
      std::cout << "Deposited " << movement.goldAmount << " gold into guild storage\n";
      printGold();
      std::cout << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kGoldGuildStorageWithdraw) {
      selfState_.addGold(movement.goldAmount);
      std::cout << "Withdrew " << movement.goldAmount << " gold from guild storage\n";
      printGold();
      std::cout << '\n';
    } else if (movement.type == packet::enums::ItemMovementType::kCosPickGold) {
      selfState_.addGold(movement.goldPickAmount);
      std::cout << "Pickpet picked " << movement.goldPickAmount << " gold\n";
      printGold();
      std::cout << '\n';
    }
  }
}

void CharacterInfoModule::abnormalInfoReceived(const packet::parsing::ParsedServerAbnormalInfo &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  for (int i=0; i<=bitNum(packet::enums::AbnormalStateFlag::kZombie); ++i) {
    selfState_.setLegacyStateEffect(state::fromBitNum(i), packet.states()[i].effectOrLevel);
  }
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStatesChanged));
}

void CharacterInfoModule::statUpdateReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  selfState_.setMaxHpMp(packet.maxHp(), packet.maxMp());
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kHpPercentChanged));
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMpPercentChanged));
}

void CharacterInfoModule::serverAgentCharacterDataReceived(const packet::parsing::ParsedServerAgentCharacterData &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  selfState_.initialize(packet.entityUniqueId(), packet.refObjId(), packet.hp(), packet.mp(), packet.masteries(), packet.skills());

  // Position
  selfState_.setPosition(packet.position());
  std::cout << "Our Ref Obj Id " << packet.refObjId() << '\n';
  std::cout << "Position: " << (packet.position().isDungeon() ? "dungeon " : "world ");
  if (packet.position().isDungeon()) {
    std::cout << '#' << (int)packet.position().dungeonId();
  } else {
    std::cout << "region (" << (int)packet.position().xSector() << ',' << (int)packet.position().zSector() << ")";
  }
  std::cout << " (" << packet.position().xOffset << ',' << packet.position().yOffset << ',' << packet.position().zOffset << ")\n";

  // State
  selfState_.setLifeState(packet.lifeState());
  selfState_.setMotionState(packet.motionState());
  selfState_.setBodyState(packet.bodyState());

  // Speed
  selfState_.setSpeed(packet.walkSpeed(), packet.runSpeed());
  selfState_.setHwanSpeed(packet.hwanSpeed());
  auto refObjId = packet.refObjId();
  selfState_.setGold(packet.gold());
  printGold();
  selfState_.setRaceAndGender(refObjId);
  const auto inventorySize = packet.inventorySize();
  const auto &inventoryItemMap = packet.inventoryItemMap();
  initializeInventory(inventorySize, inventoryItemMap);

  std::cout << "GID:" << selfState_.globalId() << ", and we have " << selfState_.hp() << " hp and " << selfState_.mp() << " mp\n";
  // eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kHpPercentChanged));
  // eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMpPercentChanged));
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kSpawned));
}

void CharacterInfoModule::initializeInventory(uint8_t inventorySize, const std::map<uint8_t, std::shared_ptr<storage::Item>> &inventoryItemMap) {
  selfState_.inventory.clear();
  selfState_.inventory.resize(inventorySize);
  // Guaranteed to have no items
  for (const auto &slotItemPtrPair : inventoryItemMap) {
    selfState_.inventory.addItem(slotItemPtrPair.first, slotItemPtrPair.second);
    printItem(slotItemPtrPair.first, slotItemPtrPair.second.get(), gameData_);
  }
}

void CharacterInfoModule::useUniversalPill() {
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

void CharacterInfoModule::usePurificationPill() {
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

void CharacterInfoModule::usePotion(PotionType potionType) {
  // We enter this funciton assuming that:
  //  1. The potion isnt on cooldown
  //  2. We have the potion
  const double hpPercentage = static_cast<double>(selfState_.hp())/(*selfState_.maxHp()); // TODO: Remove, for print only
  const double mpPercentage = static_cast<double>(selfState_.mp())/(*selfState_.maxMp()); // TODO: Remove, for print only
  printf("Healing. Hp: %4.2f%%, Mp: %4.2f%%\n", hpPercentage*100, mpPercentage*100);

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
          break;
        }
      }
    }
  }
}

void CharacterInfoModule::useItem(uint8_t slotNum, uint16_t typeData) {
  uint8_t typeId1 = (typeData >> 2) & 0b111;
  uint8_t typeId2 = (typeData >> 5) & 0b11;
  uint8_t typeId3 = (typeData >> 7) & 0b1111;
  uint8_t typeId4 = (typeData >> 11) & 0b11111;
  // TODO: Check cooldowns here
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

bool CharacterInfoModule::alreadyUsedUniversalPill() {
  if (selfState_.haveUniversalPillEventId()) {
    return true;
  }
  // Pill isnt on cooldown, but maybe we already queued a use of it
  const auto itemTypeId = makeTypeId(3, 3, 2, 6);
  return selfState_.itemIsInUsedItemQueue(itemTypeId);
}

bool CharacterInfoModule::alreadyUsedPurificationPill() {
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
  if (selfState_.purificationPillOnCooldown()) {
    return true;
  }
#endif
  const auto itemTypeId = makeTypeId(3, 3, 2, 1);
  return selfState_.itemIsInUsedItemQueue(itemTypeId);
}

bool CharacterInfoModule::alreadyUsedPotion(PotionType potionType) {
  if (potionType == PotionType::kHp) {
    if (selfState_.haveHpPotionEventId()) {
      return true;
    }
    const auto itemTypeId = makeTypeId(3, 3, 1, 1);
    return selfState_.itemIsInUsedItemQueue(itemTypeId);
  } else if (potionType == PotionType::kMp) {
    if (selfState_.haveMpPotionEventId()) {
      return true;
    }
    const auto itemTypeId = makeTypeId(3, 3, 1, 2);
    return selfState_.itemIsInUsedItemQueue(itemTypeId);
  } else if (potionType == PotionType::kVigor) {
    if (selfState_.haveVigorPotionEventId()) {
      return true;
    }
    const auto itemTypeId = makeTypeId(3, 3, 1, 3);
    return selfState_.itemIsInUsedItemQueue(itemTypeId);
  }
  // TODO: Handle other cases
  return false;
}

void CharacterInfoModule::checkIfNeedToUsePill() {
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

bool CharacterInfoModule::havePotion(PotionType potionType) {
  uint8_t typeId4;
  if (potionType == PotionType::kHp) {
    typeId4 = 1;
  } else if (potionType == PotionType::kMp) {
    typeId4 = 2;
  } else if (potionType == PotionType::kVigor) {
    typeId4 = 3;
  } else {
    std::cout << "CharacterInfoModule::havePotion: Potion type " << static_cast<int>(potionType) << " not supported\n";
    return false;
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
          return true;
        }
      }
    }
  }
  return false;
}

void CharacterInfoModule::checkIfNeedToHeal() {
  if (!selfState_.maxHp() || !selfState_.maxMp()) {
    // Dont yet know our max
    std::cout << "checkIfNeedToHeal: dont know max hp or mp\n";
    return;
  }
  if (*selfState_.maxHp() == 0) {
    // Dead, cant heal
    // TODO: Get from state update instead
    std::cout << "checkIfNeedToHeal: Dead, cant heal\n";
    return;
  }
  const double hpPercentage = static_cast<double>(selfState_.hp())/(*selfState_.maxHp());
  const double mpPercentage = static_cast<double>(selfState_.mp())/(*selfState_.maxMp());

  const auto legacyStateEffects = selfState_.legacyStateEffects();
  const bool haveZombie = (legacyStateEffects[bitNum(packet::enums::AbnormalStateFlag::kZombie)] > 0);

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

void CharacterInfoModule::entityUpdateReceived(const packet::parsing::ParsedServerHpMpUpdate &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  if (packet.entityUniqueId() != selfState_.globalId()) {
    // Not for my character, can ignore
    return;
  }
  if (packet.vitalBitmask() & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoHp)) {
    // Our HP changed
    if (selfState_.hp() != packet.newHpValue()) {
      selfState_.setHp(packet.newHpValue());
    } else {
      std::cout << "Weird, says HP changed, but it didn't\n";
    }
  }
  if (packet.vitalBitmask() & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoMp)) {
    // Our MP changed
    if (selfState_.mp() != packet.newMpValue()) {
      selfState_.setMp(packet.newMpValue());
    } else {
      std::cout << "Weird, says MP changed, but it didn't\n";
    }
  }

  if (packet.vitalBitmask() & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoAbnormal)) {
    // Our states changed
    auto stateBitmask = packet.stateBitmask();
    auto stateLevels = packet.stateLevels();
    updateStates(stateBitmask, stateLevels);
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStatesChanged));
  }

  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kHpPercentChanged));
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kMpPercentChanged));
}

int CharacterInfoModule::getGrainDelay() {
  return 4000;
}

int CharacterInfoModule::getUniversalPillDelay() {
  return 1000;
}

int CharacterInfoModule::getPurificationPillDelay() {
  // TODO: This is wrong
  return 20000;
}

void CharacterInfoModule::updateStates(uint32_t stateBitmask, const std::vector<uint8_t> &stateLevels) {
  const auto oldStateBitmask = selfState_.stateBitmask();
  uint32_t newlyReceivedStates = (oldStateBitmask ^ stateBitmask) & stateBitmask;
  uint32_t expiredStates = (oldStateBitmask ^ stateBitmask) & oldStateBitmask;
  selfState_.setStateBitmask(stateBitmask);

  int stateLevelIndex=0;
  if (newlyReceivedStates != 0) {
    // We have some new states!
    for (int bitNum=0; bitNum<32; ++bitNum) {
      const auto kBit = static_cast<uint32_t>(1) << bitNum;
      if ((newlyReceivedStates & kBit) != 0) {
        const auto kState = static_cast<packet::enums::AbnormalStateFlag>(kBit);
        if (kState <= packet::enums::AbnormalStateFlag::kZombie) {
          // Legacy state
          std::cout << "We now are " << toStr(kState) << "\n";
        } else {
          // Modern state
          selfState_.setModernStateLevel(state::fromBitNum(bitNum), stateLevels[stateLevelIndex]);
          ++stateLevelIndex;
          std::cout << "We now are under " << toStr(kState) << "\n";
        }
      }
    }
  }
  if (expiredStates != 0) {
    // We have some expired states
    for (int bitNum=0; bitNum<32; ++bitNum) {
      const auto kBit = static_cast<uint32_t>(1) << bitNum;
      if ((expiredStates & kBit) != 0) {
        const auto kState = static_cast<packet::enums::AbnormalStateFlag>(kBit);
        if (kState <= packet::enums::AbnormalStateFlag::kZombie) {
          // Legacy state
          std::cout << "We are no longer " << toStr(kState) << "\n";
        } else {
          // Modern state
          selfState_.setModernStateLevel(state::fromBitNum(bitNum), 0);
          std::cout << "We are no longer under " << toStr(kState) << "\n";
        }
      }
    }
  }
  // TODO: entity::Self
}

bool isPill(const pk2::ref::Item &itemInfo) {
  return (itemInfo.typeId1 == 3 && itemInfo.typeId2 == 3 && itemInfo.typeId3 == 2);
}

bool isUniversalPill(const pk2::ref::Item &itemInfo) {
  return (isPill(itemInfo) && itemInfo.typeId4 == 6);
}

bool isPurificationPill(const pk2::ref::Item &itemInfo) {
  return (isPill(itemInfo) && itemInfo.typeId4 == 1);
}

bool isPotion(const pk2::ref::Item &itemInfo) {
  return (itemInfo.typeId1 == 3 && itemInfo.typeId2 == 3 && itemInfo.typeId3 == 1);
}

bool isHpPotion(const pk2::ref::Item &itemInfo) {
  return (isPotion(itemInfo) && itemInfo.typeId4 == 1);
}

bool isMpPotion(const pk2::ref::Item &itemInfo) {
  return (isPotion(itemInfo) && itemInfo.typeId4 == 2);
}

bool isVigorPotion(const pk2::ref::Item &itemInfo) {
  return (isPotion(itemInfo) && itemInfo.typeId4 == 3);
}

void CharacterInfoModule::serverUseItemReceived(const packet::parsing::ParsedServerUseItem &packet) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);
  if (packet.result() == 1) {
    // Successfully used an item
    if (selfState_.inventory.hasItem(packet.slotNum())) {
      auto *itemPtr = selfState_.inventory.getItem(packet.slotNum());
      // Lets double check it's type data
      if (packet.itemData() == itemPtr->typeData()) {
        auto *expendableItemPtr = dynamic_cast<storage::ItemExpendable*>(itemPtr);
        if (expendableItemPtr != nullptr) {
          expendableItemPtr->quantity = packet.remainingCount();
          if (isHpPotion(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a potion
            if (selfState_.haveHpPotionEventId()) {
              std::cout << "Uhhhh, supposedly successfully used an hp potion when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(selfState_.getHpPotionEventId());
            }
            std::cout << "Successfully used a hpPotion\n";
            const auto hpPotionDelay = selfState_.getHpPotionDelay() + kPotionDelayBufferMs_;
            const auto hpPotionEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kHpPotionCooldownEnded), std::chrono::milliseconds(hpPotionDelay));
            selfState_.setHpPotionEventId(hpPotionEventId);
          } else if (isMpPotion(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a potion
            if (selfState_.haveMpPotionEventId()) {
              std::cout << "Uhhhh, supposedly successfully used an mp potion when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(selfState_.getMpPotionEventId());
            }
            std::cout << "Successfully used a mpPotion\n";
            const auto mpPotionDelay = selfState_.getMpPotionDelay() + kPotionDelayBufferMs_;
            const auto mpPotionEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMpPotionCooldownEnded), std::chrono::milliseconds(mpPotionDelay));
            selfState_.setMpPotionEventId(mpPotionEventId);
          } else if (isVigorPotion(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a potion
            if (selfState_.haveVigorPotionEventId()) {
              std::cout << "Uhhhh, supposedly successfully used a vigor potion when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(selfState_.getVigorPotionEventId());
            }
            std::cout << "Successfully used a vigorPotion\n";
            // TODO: Grains and regular potions have different delays, at least for Eu chars
            const auto vigorPotionDelay = selfState_.getVigorPotionDelay() + kPotionDelayBufferMs_;
            const auto vigorPotionEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kVigorPotionCooldownEnded), std::chrono::milliseconds(vigorPotionDelay));
            selfState_.setVigorPotionEventId(vigorPotionEventId);
          } else if (isUniversalPill(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a pill
            if (selfState_.haveUniversalPillEventId()) {
              std::cout << "Uhhhh, supposedly successfully used a universal pill when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(selfState_.getUniversalPillEventId());
            }
            const auto universalPillEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kUniversalPillCooldownEnded), std::chrono::milliseconds(getUniversalPillDelay()));
            selfState_.setUniversalPillEventId(universalPillEventId);
          } else if (isPurificationPill(*expendableItemPtr->itemInfo)) {
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
            // Set a timeout for how long we must wait before retrying to use a pill
            if (selfState_.purificationPillOnCooldown()) {
              std::cout << "Uhhhh, supposedly successfully used a purification pill when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(selfState_.getPurificationPillEventId());
            }
            const auto purificationPillEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kPurificationPillCooldownEnded), std::chrono::milliseconds(getPurificationPillDelay()));
            selfState_.setPurificationPillEventId(purificationPillEventId);
#endif
          }
          if (expendableItemPtr->quantity == 0) {
            std::cout << "Used the last of this item! Delete from inventory\n";
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
        std::cout << "Failed to use ";
        const auto usedItem = selfState_.getUsedItemQueueFront();
        if (usedItem.itemTypeId == makeTypeId(3, 3, 1, 1)) {
          std::cout << "hp";
        } else if (usedItem.itemTypeId == makeTypeId(3, 3, 1, 2)) {
          std::cout << "mp";
        } else if (usedItem.itemTypeId == makeTypeId(3, 3, 1, 3)) {
          std::cout << "vigor";
        }
        std::cout << " potion because there's still a cooldown, going to retry\n";
        useItem(usedItem.inventorySlotNum, usedItem.itemTypeId);
      } else if (packet.errorCode() == packet::enums::InventoryErrorCode::kCharacterDead) {
        std::cout << "Failed to use item because we're dead\n";
      } else {
        std::cout << "Unknown error while trying to use an item: " << static_cast<int>(packet.errorCode()) << '\n';
      }
    }
  }
  selfState_.popItemFromUsedItemQueueIfNotEmpty();
}

//======================================================================================================
//======================================================================================================
//======================================================================================================

void printItem(uint8_t slot, const storage::Item *item, const pk2::GameData &gameData) {
  if (item != nullptr) {
    uint16_t quantity = 1;
    const storage::ItemExpendable *itemExpendable = dynamic_cast<const storage::ItemExpendable*>(item);
    if (itemExpendable != nullptr) {
      quantity = itemExpendable->quantity;
    }
    printf("[%3d] %6d (%4d) \"%s\"\n", slot, item->refItemId, quantity, gameData.itemData().getItemById(item->refItemId).codeName128.c_str());
  } else {
    printf("[%3d] --Empty--\n", slot);
  }
}

int bitNum(packet::enums::AbnormalStateFlag flag) {
  uint32_t num = static_cast<uint32_t>(flag);
  for (int i=0; i<32; ++i) {
    if (num & (1<<i)) {
      return i;
    }
  }
  throw std::runtime_error("Trying to get bit but none set!");
}

std::string toStr(packet::enums::AbnormalStateFlag state) {
  if (state == packet::enums::AbnormalStateFlag::kNone) {
    return "none";
  } else if (state == packet::enums::AbnormalStateFlag::kFrozen) {
    return "frozen";
  } else if (state == packet::enums::AbnormalStateFlag::kFrostbitten) {
    return "frostbitten";
  } else if (state == packet::enums::AbnormalStateFlag::kShocked) {
    return "shocked";
  } else if (state == packet::enums::AbnormalStateFlag::kBurnt) {
    return "burnt";
  } else if (state == packet::enums::AbnormalStateFlag::kPoisoned) {
    return "poisoned";
  } else if (state == packet::enums::AbnormalStateFlag::kZombie) {
    return "zombie";
  } else if (state == packet::enums::AbnormalStateFlag::kSleep) {
    return "sleep";
  } else if (state == packet::enums::AbnormalStateFlag::kBind) {
    return "bind";
  } else if (state == packet::enums::AbnormalStateFlag::kDull) {
    return "dull";
  } else if (state == packet::enums::AbnormalStateFlag::kFear) {
    return "fear";
  } else if (state == packet::enums::AbnormalStateFlag::kShortSighted) {
    return "shortSighted";
  } else if (state == packet::enums::AbnormalStateFlag::kBleed) {
    return "bleed";
  } else if (state == packet::enums::AbnormalStateFlag::kPetrify) {
    return "petrify";
  } else if (state == packet::enums::AbnormalStateFlag::kDarkness) {
    return "darkness";
  } else if (state == packet::enums::AbnormalStateFlag::kStunned) {
    return "stunned";
  } else if (state == packet::enums::AbnormalStateFlag::kDisease) {
    return "disease";
  } else if (state == packet::enums::AbnormalStateFlag::kConfusion) {
    return "confusion";
  } else if (state == packet::enums::AbnormalStateFlag::kDecay) {
    return "decay";
  } else if (state == packet::enums::AbnormalStateFlag::kWeak) {
    return "weak";
  } else if (state == packet::enums::AbnormalStateFlag::kImpotent) {
    return "impotent";
  } else if (state == packet::enums::AbnormalStateFlag::kDivision) {
    return "division";
  } else if (state == packet::enums::AbnormalStateFlag::kPanic) {
    return "panic";
  } else if (state == packet::enums::AbnormalStateFlag::kCombustion) {
    return "combustion";
  } else if (state == packet::enums::AbnormalStateFlag::kEmptyBit23) {
    return "emptyBit23";
  } else if (state == packet::enums::AbnormalStateFlag::kHidden) {
    return "hidden";
  } else if (state == packet::enums::AbnormalStateFlag::kEmptyBit25) {
    return "emptyBit25";
  } else if (state == packet::enums::AbnormalStateFlag::kEmptyBit26) {
    return "emptyBit26";
  } else if (state == packet::enums::AbnormalStateFlag::kEmptyBit27) {
    return "emptyBit27";
  } else if (state == packet::enums::AbnormalStateFlag::kEmptyBit28) {
    return "emptyBit28";
  } else if (state == packet::enums::AbnormalStateFlag::kEmptyBit29) {
    return "emptyBit29";
  } else if (state == packet::enums::AbnormalStateFlag::kEmptyBit30) {
    return "emptyBit30";
  } else if (state == packet::enums::AbnormalStateFlag::kEmptyBit31) {
    return "emptyBit31";
  }
}

// TODO: Create a more elegant TypeId system
uint16_t makeTypeId(const uint16_t typeId1, const uint16_t typeId2, const uint16_t typeId3, const uint16_t typeId4) {
  return (typeId1 << 2) |
         (typeId2 << 5) |
         (typeId3 << 7) |
         (typeId4 << 11);
}

} // namespace module