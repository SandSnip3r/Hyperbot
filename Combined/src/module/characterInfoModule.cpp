#include "characterInfoModule.hpp"
#include "../packet/opcode.hpp"
#include "../packet/building/packetBuilding.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <regex>

namespace module {

int bitNum(packet::enums::AbnormalStateFlag flag);
std::string toStr(packet::enums::AbnormalStateFlag state);

//======================================================================================================
//======================================================================================================
//======================================================================================================

CharacterInfoModule::CharacterInfoModule(broker::PacketBroker &brokerSystem,
                                         broker::EventBroker &eventBroker,
                                         const packet::parsing::PacketParser &packetParser,
                                         const pk2::GameData &gameData) :
      broker_(brokerSystem),
      eventBroker_(eventBroker),
      packetParser_(packetParser),
      gameData_(gameData) {
  auto packetHandleFunction = std::bind(&CharacterInfoModule::handlePacket, this, std::placeholders::_1);
  // Client packets
  broker_.subscribeToClientPacket(packet::Opcode::CLIENT_ITEM_MOVE, packetHandleFunction);
  broker_.subscribeToClientPacket(packet::Opcode::CLIENT_CHAT, packetHandleFunction);
  // Server packets
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_AGENT_CHARACTER_INFO_DATA, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_HPMP_UPDATE, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_STATS, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_ITEM_USE, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_AGENT_ABNORMAL_INFO, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_ITEM_MOVEMENT, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_AGENT_ENTITY_GROUPSPAWN_DATA, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_SPAWN, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_DESPAWN, packetHandleFunction);

  // TODO: Save subscription ID to possibly unsubscribe in the future
  eventBroker_.subscribeToEvent(event::EventCode::kHpPotionCooldownEnded, std::bind(&CharacterInfoModule::handlePotionCooldownEnded, this, std::placeholders::_1));
  eventBroker_.subscribeToEvent(event::EventCode::kMpPotionCooldownEnded, std::bind(&CharacterInfoModule::handlePotionCooldownEnded, this, std::placeholders::_1));
  eventBroker_.subscribeToEvent(event::EventCode::kVigorPotionCooldownEnded, std::bind(&CharacterInfoModule::handlePotionCooldownEnded, this, std::placeholders::_1));
  eventBroker_.subscribeToEvent(event::EventCode::kUniversalPillCooldownEnded, std::bind(&CharacterInfoModule::handlePillCooldownEnded, this, std::placeholders::_1));
  // eventBroker_.subscribeToEvent(event::EventCode::kPurificationPillCooldownEnded, std::bind(&CharacterInfoModule::handlePillCooldownEnded, this, std::placeholders::_1));
}

void CharacterInfoModule::handlePillCooldownEnded(const std::unique_ptr<event::Event> &event) {
  std::unique_lock<std::mutex> contentionProtectionLock(contentionProtectionMutex_);
  auto eventCode = event->getEventCode();
  if (eventCode == event::EventCode::kUniversalPillCooldownEnded) {
    universalPillEventId_.reset();
    checkIfNeedToUsePill();
  }/*  else if (eventCode == event::EventCode::kPurificationPillCooldownEnded) {
    std::cout << "kPurificationPillCooldownEnded " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() << "\n";
    purificationPillEventId_.reset();
    checkIfNeedToUsePill();
  } */
}

void CharacterInfoModule::handlePotionCooldownEnded(const std::unique_ptr<event::Event> &event) {
  std::unique_lock<std::mutex> contentionProtectionLock(contentionProtectionMutex_);
  auto eventCode = event->getEventCode();
  if (eventCode == event::EventCode::kHpPotionCooldownEnded) {
    hpPotionEventId_.reset();
    checkIfNeedToHeal();
  } else if (eventCode == event::EventCode::kMpPotionCooldownEnded) {
    mpPotionEventId_.reset();
    checkIfNeedToHeal();
  } else if (eventCode == event::EventCode::kVigorPotionCooldownEnded) {
    vigorPotionEventId_.reset();
    checkIfNeedToHeal();
  }
}

bool CharacterInfoModule::handlePacket(const PacketContainer &packet) {
  std::unique_lock<std::mutex> contentionProtectionLock(contentionProtectionMutex_);
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

  auto *clientChat = dynamic_cast<packet::parsing::ParsedClientChat*>(parsedPacket.get());
  if (clientChat != nullptr) {
    std::cout << "Not handling client chat move\n";
  }


  //======================================================================================================
  //============================================Server Packets============================================
  //======================================================================================================

  auto *charData = dynamic_cast<packet::parsing::ParsedServerAgentCharacterData*>(parsedPacket.get());
  if (charData != nullptr) {
    characterInfoReceived(*charData);
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
  
  //======================================================================================================

  std::cout << "Unhandled packet subscribed to\n";
  return true;
}

void CharacterInfoModule::printObj(std::shared_ptr<packet::parsing::Object> obj) {
  switch (obj->type) {
    case packet::parsing::ObjectType::kPlayerCharacter:
      {
        auto ptr = reinterpret_cast<packet::parsing::PlayerCharacter*>(obj.get());
        printf("%7s %5d (%6f,%6f,%6f) GId:%d, name:\"%s\"\n","Player", obj->refObjId, ptr->x, ptr->y, ptr->z, ptr->gId, ptr->name.c_str());
      } 
      break;
    case packet::parsing::ObjectType::kNonplayerCharacter:
      {
        auto ptr = reinterpret_cast<packet::parsing::NonplayerCharacter*>(obj.get());
        const auto &character = gameData_.characterData().getCharacterById(obj->refObjId);
        printf("%7s %5d (%6f,%6f,%6f) \"%s\"\n","NPC", obj->refObjId, ptr->x, ptr->y, ptr->z, character.codeName128.c_str());
      } 
      break;
    case packet::parsing::ObjectType::kMonster:
      {
        auto ptr = reinterpret_cast<packet::parsing::Monster*>(obj.get());
        const auto &character = gameData_.characterData().getCharacterById(obj->refObjId);
        printf("%7s %5d (%6f,%6f,%6f) type:%d, \"%s\"\n","Monster", obj->refObjId, ptr->x, ptr->y, ptr->z, ptr->monsterRarity, character.codeName128.c_str());
      } 
      break;
    case packet::parsing::ObjectType::kItem:
      {
        auto ptr = reinterpret_cast<packet::parsing::Item*>(obj.get());
        const auto &item = gameData_.itemData().getItemById(obj->refObjId);
        printf("%7s %5d (%6f,%6f,%6f) rarity:%d, \"%s\"\n","Item", obj->refObjId, ptr->x, ptr->y, ptr->z, ptr->rarity, item.codeName128.c_str());
      } 
      break;
    case packet::parsing::ObjectType::kPortal:
      {
        auto ptr = reinterpret_cast<packet::parsing::Portal*>(obj.get());
        const auto &portal = gameData_.teleportData().getTeleportById(obj->refObjId);
        printf("%7s %5d (%6f,%6f,%6f) \"%s\"\n","Portal", obj->refObjId, ptr->x, ptr->y, ptr->z, portal.codeName128.c_str());
      }
      break;
  }
}

void CharacterInfoModule::trackObject(std::shared_ptr<packet::parsing::Object> obj) {
  objectsInRange_.emplace_back(obj);
  printf("[+++] (%5d)  ",objectsInRange_.size());
  printObj(objectsInRange_.back());
}

void CharacterInfoModule::stopTrackingObject(uint32_t gId) {
  auto it = std::find_if(objectsInRange_.begin(), objectsInRange_.end(), [gId](const std::shared_ptr<packet::parsing::Object> &objPtr) {
    return (objPtr->gId == gId);
  });
  if (it != objectsInRange_.end()) {
    auto tmp = *it;
    objectsInRange_.erase(it);
    printf("[---] (%5d)  ",objectsInRange_.size());
    printObj(tmp);
  } else {
    std::cout << "Asked to despawn something that we werent tracking\n";
  }
}

void CharacterInfoModule::serverAgentSpawnReceived(packet::parsing::ParsedServerAgentSpawn &packet) {
  trackObject(packet.object());
}

void CharacterInfoModule::serverAgentDespawnReceived(packet::parsing::ParsedServerAgentDespawn &packet) {
  stopTrackingObject(packet.gId());
}

void CharacterInfoModule::clientItemMoveReceived(const packet::parsing::ParsedClientItemMove &packet) {
  std::cout << "Handling client item move\n";
  const auto movement = packet.movement();
  if (movement.type == packet::enums::ItemMovementType::kBuyFromNPC) {
    // User is buying something from the store
    std::cout << "  bought from npc!\n";
    userPurchaseRequest_ = movement;
  }
}

void CharacterInfoModule::serverAgentGroupSpawnReceived(const packet::parsing::ParsedServerAgentGroupSpawn &packet) {
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

void CharacterInfoModule::serverItemMoveReceived(const packet::parsing::ParsedServerItemMove &packet) {
  const std::vector<packet::parsing::ItemMovement> &itemMovements = packet.itemMovements();
  std::cout << "serverItemMoveReceived, " << itemMovements.size() << '\n';
  for (const auto &movement : itemMovements) {
    if (movement.type == packet::enums::ItemMovementType::kWithinInventory) {
      std::cout << "moveItemInStorage: inventory_\n";
      inventory_.moveItemInStorage(movement.srcSlot, movement.destSlot, movement.quantity);
    } else if (movement.type == packet::enums::ItemMovementType::kWithinStorage) {
      // std::cout << "moveItemInStorage: storage_\n";
      // moveItemInStorage(storage_, movement.srcSlot, movement.destSlot, movement.quantity);
    } else if (movement.type == packet::enums::ItemMovementType::kBuyFromNPC) {
      if (userPurchaseRequest_) {
        // User purchased something, we saved this so that we can get the NPC's global Id
        std::cout << "kBuyFromNPC with ID " << userPurchaseRequest_->globalId << "\n";
        userPurchaseRequest_.reset();
      } else {
        std::cout << "kBuyFromNPC but no matching data\n";
      }
      
      // primaryItemMovement.storePageNumber = stream.Read<uint8_t>();
      // primaryItemMovement.storeSlotNumber = stream.Read<uint8_t>();
      // uint8_t stackCount = stream.Read<uint8_t>();
      // for (int i=0; i<stackCount; ++i) {
      //   // Can only happen multiple times if its an item that wont get stacked. Like equipment
      //   uint8_t inventoryDestinationSlot = stream.Read<uint8_t>();
      // }
      // primaryItemMovement.quantity = stream.Read<uint16_t>();
      // for (int i=0; i<stackCount; ++i) {
      //   auto rentInfo = parseRentInfo(stream);
      // }
    
    }
  }
}

void CharacterInfoModule::abnormalInfoReceived(const packet::parsing::ParsedServerAbnormalInfo &packet) {
  for (int i=0; i<=bitNum(packet::enums::AbnormalStateFlag::kZombie); ++i) {
    legacyStateEffects_[i] = packet.states()[i].effectOrLevel;
  }
  checkIfNeedToUsePill();
}

void CharacterInfoModule::statUpdateReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet) {
  maxHp_ = packet.maxHp();
  maxMp_ = packet.maxMp();
  checkIfNeedToHeal();
}

void CharacterInfoModule::updateRace(Race race) {
  if (race == Race::kChinese) {
    potionDelayMs_ = kChPotionDefaultDelayMs_;
  } else if (race == Race::kEuropean) {
    potionDelayMs_ = kEuPotionDefaultDelayMs_;
  }
}

void CharacterInfoModule::setRaceAndGender(uint32_t refObjId) {
  const auto &gameCharacterData = gameData_.characterData();
  if (!gameCharacterData.haveCharacterWithId(refObjId)) {
    std::cout << "Unable to determine race or gender. No \"item\" data for id: " << refObjId << '\n';
    return;
  }
  const auto &character = gameCharacterData.getCharacterById(refObjId);
  if (character.country == 0) {
    updateRace(Race::kChinese);
  } else {
    updateRace(Race::kEuropean);
  }
  if (character.charGender == 1) {
    gender_ = Gender::kMale;
  } else {
    gender_ = Gender::kFemale;
  }
}

void CharacterInfoModule::characterInfoReceived(const packet::parsing::ParsedServerAgentCharacterData &packet) {
  std::cout << "Character data received\n";
  uniqueId_ = packet.entityUniqueId();
  auto refObjId = packet.refObjId();
  setRaceAndGender(refObjId);
  hp_ = packet.hp();
  mp_ = packet.mp();
  const auto inventorySize = packet.inventorySize();
  const auto &inventoryItemMap = packet.inventoryItemMap();
  initializeInventory(inventorySize, inventoryItemMap);

  std::cout << "We are now #" << *uniqueId_ << ", and we have " << hp_ << " hp and " << mp_ << " mp\n";
  initialized_ = true;
  checkIfNeedToHeal();
}

void CharacterInfoModule::resetInventory() {
  inventory_.clear();
}

void CharacterInfoModule::initializeInventory(uint8_t inventorySize, const std::map<uint8_t, std::shared_ptr<storage::Item>> &inventoryItemMap) {
  resetInventory();
  inventory_.resize(inventorySize);
  // Guaranteed to have no items
  for (const auto &slotItemPtrPair : inventoryItemMap) {
    inventory_.addItem(slotItemPtrPair.first, slotItemPtrPair.second);
  }
}

void CharacterInfoModule::useUniversalPill() {
  // Figure out our status with the highest effect
  uint16_t ourWorstStatusEffect = *std::max_element(legacyStateEffects_.begin(), legacyStateEffects_.end());
  int32_t bestCure = 0;
  uint8_t bestOptionSlotNum;
  uint16_t bestOptionTypeData;

  for (uint8_t slotNum=0; slotNum<inventory_.size(); ++slotNum) {
    if (!inventory_.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = inventory_.getItem(slotNum);
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
  int32_t currentCureLevel = 0;
  uint8_t bestOptionSlotNum;
  uint16_t bestOptionTypeData;

  for (uint8_t slotNum=0; slotNum<inventory_.size(); ++slotNum) {
    if (!inventory_.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = inventory_.getItem(slotNum);
    const storage::ItemExpendable *item;
    if ((item = dynamic_cast<const storage::ItemExpendable*>(itemPtr)) != nullptr) {
      // Expendable item
      if (item->itemInfo->typeId1 == 3 && item->itemInfo->typeId2 == 3 && item->itemInfo->typeId3 == 2 && item->itemInfo->typeId4 == 1) {
        // Purification pill
        const auto pillCureStateBitmask = item->itemInfo->param1;
        const auto curableStatesWeHave = (pillCureStateBitmask & stateBitmask_);
        if (curableStatesWeHave > 0) {
          // This pill will cure at least some of the type of state(s) that we have
          const auto pillTreatmentLevel = item->itemInfo->param2;
          if (pillTreatmentLevel != currentCureLevel) {
            std::vector<uint8_t> stateLevels;
            for (uint32_t bitNum=0; bitNum<32; ++bitNum) {
              const auto bit = 1 << bitNum;
              if (curableStatesWeHave & bit) {
                stateLevels.push_back(modernStateLevel_[bitNum]);
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
  const double hpPercentage = static_cast<double>(hp_)/(*maxHp_); // TODO: Remove, for print only
  const double mpPercentage = static_cast<double>(mp_)/(*maxMp_); // TODO: Remove, for print only
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
  for (uint8_t slotNum=0; slotNum<inventory_.size(); ++slotNum) {
    if (!inventory_.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = inventory_.getItem(slotNum);
    const storage::ItemExpendable *item;
    if ((item = dynamic_cast<const storage::ItemExpendable*>(itemPtr)) != nullptr) {
      // Expendable item
      if (item->itemInfo->typeId1 == 3 && item->itemInfo->typeId2 == 3 && item->itemInfo->typeId3 == 1 && item->itemInfo->typeId4 == typeId4) {
        if (item->itemInfo->param2 == 0 && item->itemInfo->param4 == 0) {
          // Avoid vigors
          useItem(slotNum, itemPtr->typeData());
          break;
        }
      }
    }
  }
}

void CharacterInfoModule::useItem(uint8_t slotNum, uint16_t typeData) {
  auto useItemPacket = packet::building::ClientUseItemBuilder(slotNum, typeData).packet();
  broker_.injectPacket(useItemPacket, PacketContainer::Direction::kClientToServer);
  usedItemQueue_.emplace_back(slotNum, typeData);
}

bool CharacterInfoModule::alreadyUsedUniversalPill() {
  bool used = false;
  for (const auto &usedItem : usedItemQueue_) {
    if (usedItem.itemData & (static_cast<uint16_t>(3) << 2) &&
        usedItem.itemData & (static_cast<uint16_t>(3) << 5) &&
        usedItem.itemData & (static_cast<uint16_t>(2) << 7) &&
        usedItem.itemData & (static_cast<uint16_t>(6) << 11)) {
      used = true;
      break;
    }
  }
  return (universalPillEventId_ || used); 
}

bool CharacterInfoModule::alreadyUsedPurificationPill() {
  bool used = false;
  for (const auto &usedItem : usedItemQueue_) {
    if (usedItem.itemData & (static_cast<uint16_t>(3) << 2) &&
        usedItem.itemData & (static_cast<uint16_t>(3) << 5) &&
        usedItem.itemData & (static_cast<uint16_t>(2) << 7) &&
        usedItem.itemData & (static_cast<uint16_t>(1) << 11)) {
      used = true;
      break;
    }
  }
  return (/* purificationPillEventId_ || */ used);
}

bool CharacterInfoModule::alreadyUsedPotion(PotionType potionType) {
  if (potionType == PotionType::kHp) {
    bool used = false;
    for (const auto &usedItem : usedItemQueue_) {
      if (usedItem.itemData & (static_cast<uint16_t>(3) << 2) &&
          usedItem.itemData & (static_cast<uint16_t>(3) << 5) &&
          usedItem.itemData & (static_cast<uint16_t>(1) << 7) &&
          usedItem.itemData & (static_cast<uint16_t>(1) << 11)) {
        used = true;
        break;
      }
    }
    return (hpPotionEventId_ || used);
  } else if (potionType == PotionType::kMp) {
    bool used = false;
    for (const auto &usedItem : usedItemQueue_) {
      if (usedItem.itemData & (static_cast<uint16_t>(3) << 2) &&
          usedItem.itemData & (static_cast<uint16_t>(3) << 5) &&
          usedItem.itemData & (static_cast<uint16_t>(1) << 7) &&
          usedItem.itemData & (static_cast<uint16_t>(2) << 11)) {
        used = true;
        break;
      }
    }
    return (mpPotionEventId_ || used);
  } else if (potionType == PotionType::kVigor) {
    bool used = false;
    for (const auto &usedItem : usedItemQueue_) {
      if (usedItem.itemData & (static_cast<uint16_t>(3) << 2) &&
          usedItem.itemData & (static_cast<uint16_t>(3) << 5) &&
          usedItem.itemData & (static_cast<uint16_t>(1) << 7) &&
          usedItem.itemData & (static_cast<uint16_t>(3) << 11)) {
        used = true;
        break;
      }
    }
    return (vigorPotionEventId_ || used);
  }
  // TODO: Handle other cases
  return false;
}

void CharacterInfoModule::checkIfNeedToUsePill() {
  if (std::any_of(legacyStateEffects_.begin(), legacyStateEffects_.end(), [](const uint16_t effect){ return effect > 0; })) {
    // Need to use a universal pill
    if (!alreadyUsedUniversalPill()) {
      useUniversalPill();
    }
  }
  if (std::any_of(modernStateLevel_.begin(), modernStateLevel_.end(), [](const uint8_t level){ return level > 0; })) {
    // Need to use purification pill
    if (!alreadyUsedPurificationPill()) {
      usePurificationPill();
    }
  }
}

void CharacterInfoModule::checkIfNeedToHeal() {
  if (!maxHp_ || !maxMp_) {
    // Dont yet know our max
    std::cout << "checkIfNeedToHeal: dont know max hp or mp\n";
    return;
  }
  if (*maxHp_ == 0) {
    // Either uninitialized or dead. Cant heal in either case probably
    std::cout << "checkIfNeedToHeal: Either uninitialized or dead. Cant heal in either case probably\n";
    // TODO: Figure out
    return;
  }
  const double hpPercentage = static_cast<double>(hp_)/(*maxHp_);
  const double mpPercentage = static_cast<double>(mp_)/(*maxMp_);

  const bool haveZombie = (legacyStateEffects_[bitNum(packet::enums::AbnormalStateFlag::kZombie)] > 0);

  if ((!haveZombie && hpPercentage <= kHpThreshold_) && mpPercentage <= kMpThreshold_) {
    if (!alreadyUsedPotion(PotionType::kVigor)) {
      usePotion(PotionType::kVigor);
    }
  } else if ((!haveZombie && hpPercentage <= kHpThreshold_)) {
    if (!alreadyUsedPotion(PotionType::kHp)) {
      usePotion(PotionType::kHp);
    } else if (hpPercentage < kVigorThreshold_ && !alreadyUsedPotion(PotionType::kVigor)) {
      usePotion(PotionType::kVigor);
    }
  } else if (mpPercentage <= kMpThreshold_) {
    if (!alreadyUsedPotion(PotionType::kMp)) {
      usePotion(PotionType::kMp);
    } else if (mpPercentage < kVigorThreshold_ && !alreadyUsedPotion(PotionType::kVigor)) {
      usePotion(PotionType::kVigor);
    }
  }
}

void CharacterInfoModule::entityUpdateReceived(const packet::parsing::ParsedServerHpMpUpdate &packet) {
  if (uniqueId_ && packet.entityUniqueId() != *uniqueId_) {
    // Not for my character, can ignore
    return;
  }
  if (packet.vitalBitmask() & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoHp)) {
    // Our HP changed
    if (hp_ != packet.newHpValue()) {
      hp_ = packet.newHpValue();
    } else {
      std::cout << "Weird, says HP changed, but it didn't\n";
    }
  }
  if (packet.vitalBitmask() & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoMp)) {
    // Our MP changed
    if (mp_ != packet.newMpValue()) {
      mp_ = packet.newMpValue();
    } else {
      std::cout << "Weird, says MP changed, but it didn't\n";
    }
  }

  if (packet.vitalBitmask() & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoAbnormal)) {
    // Our states changed
    auto stateBitmask = packet.stateBitmask();
    auto stateLevels = packet.stateLevels();
    updateStates(stateBitmask, stateLevels);
    checkIfNeedToUsePill();
  }

  checkIfNeedToHeal();
}

int CharacterInfoModule::getHpPotionDelay() {
  const bool havePanic = (modernStateLevel_[bitNum(packet::enums::AbnormalStateFlag::kPanic)] > 0);
  int delay = potionDelayMs_ + kPotionDelayBufferMs_;
  if (havePanic) {
    delay += 4000;
  }
  return delay;
}

int CharacterInfoModule::getMpPotionDelay() {
  const bool haveCombustion = (modernStateLevel_[bitNum(packet::enums::AbnormalStateFlag::kCombustion)] > 0);
  int delay = potionDelayMs_ + kPotionDelayBufferMs_;
  if (haveCombustion) {
    delay += 4000;
  }
  return delay;
}

int CharacterInfoModule::getVigorPotionDelay() {
  return potionDelayMs_ + kPotionDelayBufferMs_;
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
  uint32_t newlyReceivedStates = (stateBitmask_ ^ stateBitmask) & stateBitmask;
  uint32_t expiredStates = (stateBitmask_ ^ stateBitmask) & stateBitmask_;
  stateBitmask_ = stateBitmask;

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
          modernStateLevel_[bitNum] = stateLevels[stateLevelIndex];
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
          modernStateLevel_[bitNum] = 0;
          std::cout << "We are no longer under " << toStr(kState) << "\n";
        }
      }
    }
  }
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
  if (packet.result() == 1) {
    // Successfully used an item
    if (inventory_.hasItem(packet.slotNum())) {
      auto *itemPtr = inventory_.getItem(packet.slotNum());
      // Lets double check it's type data
      if (packet.itemData() == itemPtr->typeData()) {
        auto *expendableItemPtr = dynamic_cast<storage::ItemExpendable*>(itemPtr);
        if (expendableItemPtr != nullptr) {
          expendableItemPtr->stackCount = packet.remainingCount();
          if (isHpPotion(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a potion
            if (hpPotionEventId_) {
              std::cout << "Uhhhh, supposedly successfully used an hp potion when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(*hpPotionEventId_);
            }
            hpPotionEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kHpPotionCooldownEnded), std::chrono::milliseconds(getHpPotionDelay()));
          } else if (isMpPotion(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a potion
            if (mpPotionEventId_) {
              std::cout << "Uhhhh, supposedly successfully used an mp potion when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(*mpPotionEventId_);
            }
            mpPotionEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMpPotionCooldownEnded), std::chrono::milliseconds(getMpPotionDelay()));
          } else if (isVigorPotion(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a potion
            if (vigorPotionEventId_) {
              std::cout << "Uhhhh, supposedly successfully used a vigor potion when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(*vigorPotionEventId_);
            }
            vigorPotionEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kVigorPotionCooldownEnded), std::chrono::milliseconds(getVigorPotionDelay()));
          } else if (isUniversalPill(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a pill
            if (universalPillEventId_) {
              std::cout << "Uhhhh, supposedly successfully used a universal pill when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(*universalPillEventId_);
            }
            universalPillEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kUniversalPillCooldownEnded), std::chrono::milliseconds(getUniversalPillDelay()));
          } else if (isPurificationPill(*expendableItemPtr->itemInfo)) {
            // Set a timeout for how long we must wait before retrying to use a pill
            /* if (purificationPillEventId_) {
              std::cout << "Uhhhh, supposedly successfully used a purification pill when there's still a cooldown... Cancelling timer\n";
              eventBroker_.cancelDelayedEvent(*purificationPillEventId_);
            } */
            // purificationPillEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kPurificationPillCooldownEnded), std::chrono::milliseconds(getPurificationPillDelay()));
          }
          if (expendableItemPtr->stackCount == 0) {
            std::cout << "Used the last of this item! Delete from inventory\n";
            // TODO: Instead, delete the item upon receiving server_item_movement in the case DEL_ITEM_BY_SERVER
            inventory_.deleteItem(packet.slotNum());
          }
        }
      }
    }
  } else {
    // Failed to use item
    if (!usedItemQueue_.empty()) {
      // This was an item that we tried to use
      if (packet.errorCode() == packet::enums::InventoryErrorCode::kWaitForReuseDelay) {
        std::cout << "Failed to use item because there's still a cooldown, going to retry\n";
        // TODO: When we start tracking items moving in the invetory, we'll need to somehow update usedItemQueue_
        const auto usedItem = usedItemQueue_.front();
        useItem(usedItem.slotNum, usedItem.itemData);
      }
    }
  }
  if (!usedItemQueue_.empty()) {
    usedItemQueue_.pop_front();
  }
}

//======================================================================================================
//======================================================================================================
//======================================================================================================

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

} // namespace module