#include "parsedPacket.hpp"

#include <iostream>

namespace packet::parsing {

//=========================================================================================================================================================

ParsedPacket::ParsedPacket(const PacketContainer &packet) : opcode_(static_cast<Opcode>(packet.opcode)) {}

Opcode ParsedPacket::opcode() const {
  return opcode_;
}

ParsedPacket::~ParsedPacket() {}

//=========================================================================================================================================================

ParsedUnknown::ParsedUnknown(const PacketContainer &packet) : ParsedPacket(packet) {}

//=========================================================================================================================================================

ParsedServerHpMpUpdate::ParsedServerHpMpUpdate(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;

  entityUniqueId_ = stream.Read<uint32_t>();
  updateFlag_ = static_cast<packet_enums::UpdateFlag>(stream.Read<uint16_t>());
  vitalBitmask_ = stream.Read<uint8_t>();

  if (vitalBitmask_ & static_cast<uint8_t>(packet_enums::VitalInfoFlag::kVitalInfoHp)) {
    newHpValue_ = stream.Read<uint32_t>();
  }

  if (vitalBitmask_ & static_cast<uint8_t>(packet_enums::VitalInfoFlag::kVitalInfoMp)) {
    newMpValue_ = stream.Read<uint32_t>();
  }

  if (vitalBitmask_ & static_cast<uint8_t>(packet_enums::VitalInfoFlag::kVitalInfoHgp)) {
    newHgpValue_ = stream.Read<uint16_t>();
  }

  if (vitalBitmask_ & static_cast<uint8_t>(packet_enums::VitalInfoFlag::kVitalInfoAbnormal)) {
    stateBitmask_ = stream.Read<uint32_t>();
    for (uint32_t i=0; i<32; ++i) {
      const auto bit = (1 << i);
      if (bit > static_cast<uint32_t>(packet_enums::AbnormalStateFlag::kZombie) && stateBitmask_ & bit) {
        stateLevels_.push_back(stream.Read<uint8_t>());
      }
    }
  }
}

uint32_t ParsedServerHpMpUpdate::entityUniqueId() const {
  return entityUniqueId_;
}

packet_enums::UpdateFlag ParsedServerHpMpUpdate::updateFlag() const {
  return updateFlag_;
}

uint8_t ParsedServerHpMpUpdate::vitalBitmask() const {
  return vitalBitmask_;
}

uint32_t ParsedServerHpMpUpdate::newHpValue() const {
  return newHpValue_;
}

uint32_t ParsedServerHpMpUpdate::newMpValue() const {
  return newMpValue_;
}

uint16_t ParsedServerHpMpUpdate::newHgpValue() const {
  return newHgpValue_;
}

uint32_t ParsedServerHpMpUpdate::stateBitmask() const {
  return stateBitmask_;
}

const std::vector<uint8_t>& ParsedServerHpMpUpdate::stateLevels() const {
  return stateLevels_;
}

//=========================================================================================================================================================

uint32_t ParsedServerAbnormalInfo::stateBitmask() const {
  return stateBitmask_;
}

const std::array<PacketInnerStructures::vitals::AbnormalState, 32>& ParsedServerAbnormalInfo::states() const {
  return states_;
}

ParsedServerAbnormalInfo::ParsedServerAbnormalInfo(const PacketContainer &packet) : ParsedPacket(packet) { 
  StreamUtility stream = packet.data;
  stateBitmask_ = stream.Read<uint32_t>();
  for (uint32_t i=0; i<32; ++i) {
    const auto bit = (1 << i);
    if (stateBitmask_ & bit) {
      auto &state = states_[i];
      state.totalTime = stream.Read<uint32_t>();
      state.timeElapsed = stream.Read<uint16_t>();
      if (bit <= static_cast<uint32_t>(packet_enums::AbnormalStateFlag::kZombie)) {
        // Legacy states
        state.effectOrLevel = stream.Read<uint16_t>();
      } else {
        // Modern states
        state.effectOrLevel = stream.Read<uint16_t>();
      }
    }
  }
 }

//=========================================================================================================================================================

uint8_t ParsedServerUseItem::result() const {
  return result_;
}

uint8_t ParsedServerUseItem::slotNum() const {
  return slotNum_;
}

uint16_t ParsedServerUseItem::remainingCount() const {
  return remainingCount_;
}

uint16_t ParsedServerUseItem::itemData() const {
  return itemData_;
}

packet_enums::InventoryErrorCode ParsedServerUseItem::errorCode() const {
  return errorCode_;
}

ParsedServerUseItem::ParsedServerUseItem(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;

  result_ = stream.Read<uint8_t>();
  if (result_ == 1) {
    // Success
    slotNum_ = stream.Read<uint8_t>();
    remainingCount_ = stream.Read<uint16_t>();
    itemData_ = stream.Read<uint16_t>();
  } else {
    errorCode_ = static_cast<packet_enums::InventoryErrorCode>(stream.Read<uint16_t>());
  }
}

//=========================================================================================================================================================

RentInfo parseRentInfo(StreamUtility &stream) {
  RentInfo rentInfo;
  rentInfo.rentType = stream.Read<uint32_t>();
  
  if (rentInfo.rentType == 1) {
    rentInfo.canDelete = stream.Read<uint16_t>();
    rentInfo.periodBeginTime = stream.Read<uint32_t>();
    rentInfo.periodEndTime = stream.Read<uint32_t>();
  } else if (rentInfo.rentType == 2) {
    rentInfo.canDelete = stream.Read<uint16_t>();
    rentInfo.canRecharge = stream.Read<uint16_t>();
    rentInfo.meterRateTime = stream.Read<uint32_t>();
  } else if (rentInfo.rentType == 3) {
    rentInfo.canDelete = stream.Read<uint16_t>();
    rentInfo.canRecharge = stream.Read<uint16_t>();
    rentInfo.periodBeginTime = stream.Read<uint32_t>();
    rentInfo.periodEndTime = stream.Read<uint32_t>();
    rentInfo.packingTime = stream.Read<uint32_t>();
  }
  return rentInfo;
}

const std::vector<ItemMovement>& ParsedServerItemMove::itemMovements() const {
  return itemMovements_;
}

ParsedServerItemMove::ParsedServerItemMove(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint8_t result_ = stream.Read<uint8_t>();
  std::cout << "ParsedServerItemMove " << (int)result_ << '\n';
  if (result_ == 1) {
    // Success
    ItemMovement primaryItemMovement;
    primaryItemMovement.type = static_cast<packet_enums::ItemMovementType>(stream.Read<uint8_t>());
    if (primaryItemMovement.type == packet_enums::ItemMovementType::kWithinInventory ||
        primaryItemMovement.type == packet_enums::ItemMovementType::kAvatarToInventory ||
        primaryItemMovement.type == packet_enums::ItemMovementType::kInventoryToAvatar) {
      std::cout << "kWithinInventory\n";
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>(); // Seems to be 0 when equiping or unequiping gear/avatars

      uint8_t secondaryMovementCount = stream.Read<uint8_t>();
      // While moving things around inside our inventory, there's a possibility that more items get moved too
      //  Like when we remove our dress, the accessory is forcefully removed
      for (int i=0; i<secondaryMovementCount; ++i) {
        std::cout << "Secondary!\n";
        ItemMovement secondaryItemMovement;
        secondaryItemMovement.type = static_cast<packet_enums::ItemMovementType>(stream.Read<uint8_t>());
        // TODO: We assume that it will always be an inventory movement. However, it could
        //  technically be any kind of movement with the same data structure (ex. withinStorage)
        secondaryItemMovement.srcSlot = stream.Read<uint8_t>();
        secondaryItemMovement.destSlot = stream.Read<uint8_t>();
        secondaryItemMovement.quantity = stream.Read<uint16_t>(); // Seems to be 0 when equiping or unequiping gear/avatars
        itemMovements_.push_back(secondaryItemMovement);
      }
    } else if (primaryItemMovement.type == packet_enums::ItemMovementType::kInventoryToStorage ||
               primaryItemMovement.type == packet_enums::ItemMovementType::kStorageToInventory ||
               primaryItemMovement.type == packet_enums::ItemMovementType::kInventoryToGuildStorage ||
               primaryItemMovement.type == packet_enums::ItemMovementType::kGuildStorageToInventory) {
      std::cout << "kInventoryToStorage, kStorageToInventory, kInventoryToGuildStorage, kGuildStorageToInventory\n";
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
    } else if (primaryItemMovement.type == packet_enums::ItemMovementType::kInventoryToStorage) {
      std::cout << "kInventoryToStorage\n";
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>();
    } else if (primaryItemMovement.type == packet_enums::ItemMovementType::kGoldDrop ||
               primaryItemMovement.type == packet_enums::ItemMovementType::kGoldStorageWithdraw ||
               primaryItemMovement.type == packet_enums::ItemMovementType::kGoldStorageDeposit ||
               primaryItemMovement.type == packet_enums::ItemMovementType::kGoldGuildStorageDeposit ||
               primaryItemMovement.type == packet_enums::ItemMovementType::kGoldGuildStorageWithdraw) {
      std::cout << "kGoldDrop, kGoldStorageWithdraw, kGoldStorageDeposit, kGoldGuildStorageDeposit, kGoldGuildStorageWithdraw\n";
      primaryItemMovement.goldAmount = stream.Read<uint64_t>();
    } else if (primaryItemMovement.type == packet_enums::ItemMovementType::kGoldPick) {
      std::cout << "kGoldPick\n";
      primaryItemMovement.destSlot = stream.Read<uint8_t>(); // Gold slot, always 0xFE
      primaryItemMovement.goldPickAmount = stream.Read<uint32_t>();
    } else if (primaryItemMovement.type == packet_enums::ItemMovementType::kWithinCos) {
      std::cout << "kWithinCos\n";
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // COS global ID
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>();
    } else if (primaryItemMovement.type == packet_enums::ItemMovementType::kCosToInventory ||
               primaryItemMovement.type == packet_enums::ItemMovementType::kInventoryToCos) {
      std::cout << "kCosToInventory, kInventoryToCos\n";
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // COS global ID
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
    } else if (primaryItemMovement.type == packet_enums::ItemMovementType::kWithinStorage ||
               primaryItemMovement.type == packet_enums::ItemMovementType::kWithinGuildStorage) {
      std::cout << "kWithinStorage, kWithinGuildStorage\n";
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>();
    } else if (primaryItemMovement.type == packet_enums::ItemMovementType::kBuyFromNPC) {
      std::cout << "kBuyFromNPC\n";
      primaryItemMovement.storePageNumber = stream.Read<uint8_t>();
      primaryItemMovement.storeSlotNumber = stream.Read<uint8_t>();
      uint8_t stackCount = stream.Read<uint8_t>();
      for (int i=0; i<stackCount; ++i) {
        // Can only happen multiple times if its an item that wont get stacked. Like equipment
        uint8_t inventoryDestinationSlot = stream.Read<uint8_t>();
      }
      primaryItemMovement.quantity = stream.Read<uint16_t>();
      for (int i=0; i<stackCount; ++i) {
        auto rentInfo = parseRentInfo(stream);
      }
    } else if (primaryItemMovement.type == packet_enums::ItemMovementType::kCosPickGold) {
      std::cout << "kCosPickGold\n";
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // COS global ID
      primaryItemMovement.destSlot = stream.Read<uint8_t>(); // Gold slot, always 0xFE
      primaryItemMovement.quantity = stream.Read<uint32_t>();
    } else if (primaryItemMovement.type == packet_enums::ItemMovementType::kSellToNPC) {
      std::cout << "kSellToNPC\n";
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>();
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // NPC global ID
      primaryItemMovement.buybackStackSize = stream.Read<uint8_t>();
    } else if (primaryItemMovement.type == packet_enums::ItemMovementType::kBuyback) {
      std::cout << "kBuyback\n";
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.srcSlot = stream.Read<uint8_t>(); // Shop buyback slot, left is max(buybackStackSize-1), right is 0
      primaryItemMovement.quantity = stream.Read<uint16_t>();
    } else {
      std::cout << "Unhandled item movement case! Type: " << static_cast<int>(primaryItemMovement.type) << '\n';
    }
    if (!itemMovements_.empty()) {
      // There were secondary item movements added, place the primary item movement at the beginning of the list
      itemMovements_.insert(itemMovements_.begin(), primaryItemMovement);
    } else {
      itemMovements_.push_back(primaryItemMovement);
    }
  } else {
    std::cout << "Item movement failed! Dumping data\n";
    std::cout << DumpToString(stream) << '\n';
  }
}
// TODO: Try to inject a buy packet that buys more than 1 stack of a stackable item
//  Invesitage what it does to the kBuyFromNPC data

//=========================================================================================================================================================

uint32_t ParsedServerAgentCharacterUpdateStats::maxHp() const {
  return maxHp_;
}

uint32_t ParsedServerAgentCharacterUpdateStats::maxMp() const {
  return maxMp_;
}

ParsedServerAgentCharacterUpdateStats::ParsedServerAgentCharacterUpdateStats(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;

  uint32_t phyAtkMin = stream.Read<uint32_t>();
  uint32_t phyAtkMax = stream.Read<uint32_t>();
  uint32_t magAtkMin = stream.Read<uint32_t>();
  uint32_t magAtkMax = stream.Read<uint32_t>();
  uint16_t phyDef = stream.Read<uint16_t>();
  uint16_t magDef = stream.Read<uint16_t>();
  uint16_t hitRate = stream.Read<uint16_t>();
  uint16_t parryRate = stream.Read<uint16_t>();
  maxHp_ = stream.Read<uint32_t>();
  maxMp_ = stream.Read<uint32_t>();
  uint16_t strPts = stream.Read<uint16_t>();
  uint16_t intPts = stream.Read<uint16_t>();
}

//=========================================================================================================================================================

uint32_t ParsedServerAgentCharacterData::refObjId() const {
  return refObjId_;
}

uint32_t ParsedServerAgentCharacterData::entityUniqueId() const {
  return entityUniqueId_;
}

uint32_t ParsedServerAgentCharacterData::hp() const {
  return hp_;
}

uint32_t ParsedServerAgentCharacterData::mp() const {
  return mp_;
}

uint8_t ParsedServerAgentCharacterData::inventorySize() const {
  return inventorySize_;
}

const std::map<uint8_t, std::shared_ptr<item::Item>>& ParsedServerAgentCharacterData::inventoryItemMap() const {
  return inventoryItemMap_;
}

void parseItem(item::ItemEquipment &item, StreamUtility &stream) {
  item.optLevel = stream.Read<uint8_t>();
  item.variance = stream.Read<uint64_t>();
  item.durability = stream.Read<uint32_t>();
  uint8_t magParamNum = stream.Read<uint8_t>();
  for (int paramIndex=0; paramIndex<magParamNum; ++paramIndex) {
    item.magicParams.emplace_back();
    auto &magParam = item.magicParams.back();
    magParam.type = stream.Read<uint32_t>();
    magParam.value = stream.Read<uint32_t>();
  }
  
  uint8_t bindingOptionType1 = stream.Read<uint8_t>(); // Weird useless byte (to represent Socket)
  uint8_t bindingOptionCount1 = stream.Read<uint8_t>();
  for (int bindingOptionIndex=0; bindingOptionIndex<bindingOptionCount1; ++bindingOptionIndex) {
    item.socketOptions.emplace_back();
    auto &socketOption = item.socketOptions.back();
    socketOption.slot = stream.Read<uint8_t>();
    socketOption.id = stream.Read<uint32_t>();
    socketOption.nParam1 = stream.Read<uint32_t>();
  }
  
  uint8_t bindingOptionType2 = stream.Read<uint8_t>(); // Weird useless byte (to represent Advanced Elixir)
  uint8_t bindingOptionCount2 = stream.Read<uint8_t>();
  for (int bindingOptionIndex=0; bindingOptionIndex<bindingOptionCount2; ++bindingOptionIndex) {
    item.advancedElixirOptions.emplace_back();
    auto &advancedElixirOption = item.advancedElixirOptions.back();
    advancedElixirOption.slot = stream.Read<uint8_t>();
    advancedElixirOption.id = stream.Read<uint32_t>();
    advancedElixirOption.optValue = stream.Read<uint32_t>();
  }
}

void parseItemCosSummoner(item::ItemCosGrowthSummoner *cosSummoner, StreamUtility &stream) {  
  cosSummoner->lifeState = static_cast<item::CosLifeState>(stream.Read<uint8_t>());
  if (cosSummoner->lifeState != item::CosLifeState::kInactive) {
    cosSummoner->refObjID = stream.Read<uint32_t>();
    uint16_t nameLength = stream.Read<uint16_t>();
    cosSummoner->name = stream.Read_Ascii(nameLength);

    // Special case for ability pets
    if (cosSummoner->type == item::ItemType::kItemCosAbilitySummoner) {
      auto *cosAbilitySummoner = dynamic_cast<item::ItemCosAbilitySummoner*>(cosSummoner);
      if (cosAbilitySummoner != nullptr) {
        cosAbilitySummoner->secondsToRentEndTime = stream.Read<uint32_t>();
      } else {
        throw std::runtime_error("Trying to cast Item to type ItemCosAbilitySummoner failed");
      }
    }

    uint8_t timedJobCount = stream.Read<uint8_t>();
    for (int jobNum=0; jobNum<timedJobCount; ++jobNum) {
      cosSummoner->jobs.emplace_back();
      auto &job = cosSummoner->jobs.back();
      job.category = stream.Read<uint8_t>();
      job.jobId = stream.Read<uint32_t>();
      job.timeToKeep = stream.Read<uint32_t>();
      if (job.category == 5) {
        job.data1 = stream.Read<uint32_t>();
        job.data2 = stream.Read<uint8_t>();
      }
    }
  }
}

void parseItem(item::ItemCosGrowthSummoner &item, StreamUtility &stream) {
  parseItemCosSummoner(&item, stream);
}

void parseItem(item::ItemCosAbilitySummoner &item, StreamUtility &stream) {
  parseItemCosSummoner(&item, stream);
}

void parseItem(item::ItemMonsterCapsule &item, StreamUtility &stream) {
  item.refObjID = stream.Read<uint32_t>();
}

void parseItem(item::ItemStorage &item, StreamUtility &stream) {
  item.quantity = stream.Read<uint32_t>();
}

void parseItem(item::ItemExpendable &item, StreamUtility &stream) {
  item.stackCount = stream.Read<uint16_t>();
}

void parseItem(item::ItemStone &item, StreamUtility &stream) {
  parseItem(*dynamic_cast<item::ItemExpendable*>(&item), stream);

  item.attributeAssimilationProbability = stream.Read<uint8_t>();
}

void parseItem(item::ItemMagicPop &item, StreamUtility &stream) {
  parseItem(*dynamic_cast<item::ItemExpendable*>(&item), stream);

  uint8_t magParamCount = stream.Read<uint8_t>();
  for (int paramIndex=0; paramIndex<magParamCount; ++paramIndex) {
    item.magicParams.emplace_back();
    auto &magicParam = item.magicParams.back();
    magicParam.type = stream.Read<uint32_t>();
    magicParam.value = stream.Read<uint32_t>();
  }
}

void parseItem(item::Item *item, StreamUtility &stream) {
  using namespace item;
  if (item->type == ItemType::kItemEquipment) {
    auto *equipment = dynamic_cast<ItemEquipment*>(item);
    if (equipment != nullptr) {
      parseItem(*equipment, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemEquipment failed");
    }
  } else if (item->type == ItemType::kItemCosAbilitySummoner) {
    auto *cosAbilitySummoner = dynamic_cast<ItemCosAbilitySummoner*>(item);
    if (cosAbilitySummoner != nullptr) {
      parseItem(*cosAbilitySummoner, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemCosAbilitySummoner failed");
    }
  } else if (item->type == ItemType::kItemCosGrowthSummoner) {
    auto *cosGrowthSummoner = dynamic_cast<ItemCosGrowthSummoner*>(item);
    if (cosGrowthSummoner != nullptr) {
      parseItem(*cosGrowthSummoner, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemCosGrowthSummoner failed");
    }
  } else if (item->type == ItemType::kItemMonsterCapsule) {
    auto *monsterCapsule = dynamic_cast<ItemMonsterCapsule*>(item);
    if (monsterCapsule != nullptr) {
      parseItem(*monsterCapsule, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemMonsterCapsule failed");
    }
  } else if (item->type == ItemType::kItemStorage) {
    auto *storage = dynamic_cast<ItemStorage*>(item);
    if (storage != nullptr) {
      parseItem(*storage, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemStorage failed");
    }
  } else if (item->type == ItemType::kItemStone) {
    auto *stone = dynamic_cast<ItemStone*>(item);
    if (stone != nullptr) {
      parseItem(*stone, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemStone failed");
    }
  } else if (item->type == ItemType::kItemMagicPop) {
    auto *magicPop = dynamic_cast<ItemMagicPop*>(item);
    if (magicPop != nullptr) {
      parseItem(*magicPop, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemMagicPop failed");
    }
  } else if (item->type == ItemType::kItemExpendable) {
    auto *expendable = dynamic_cast<ItemExpendable*>(item);
    if (expendable != nullptr) {
      parseItem(*expendable, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemExpendable failed");
    }
  }
}

item::Item* newItemByTypeData(const pk2::media::Item &item) {
  using namespace item;

  if (item.typeId1 == 3) {
    if (item.typeId2 == 1) {
      // CGItemEquip
      return new ItemEquipment();
    } else if (item.typeId2 == 2) {
      if (item.typeId3 == 1) {                                
        // CGItemCOSSummoner
        if (item.typeId4 == 2) {
          return new ItemCosAbilitySummoner();
        } else {
          return new ItemCosGrowthSummoner();
        }
      } else if (item.typeId3 == 2) {
        // CGItemMonsterCapsule (rogue mask)
        return new ItemMonsterCapsule();
      } else if (item.typeId3 == 3) {
        // CGItemStorage
        return new ItemStorage();
      }
    } else if (item.typeId2 == 3) {
      // CGItemExpendable
      if (item.typeId3 == 11) {
        if (item.typeId4 == 1 || item.typeId4 == 2) {
          // MAGICSTONE, ATTRSTONE
          return new ItemStone();
        }
      } else if (item.typeId3 == 14 && item.typeId4 == 2) {
        // Magic pop
        return new ItemMagicPop();
      }
      // Other expendable
      return new ItemExpendable();
    }
  }
  return nullptr;
}

ParsedServerAgentCharacterData::ParsedServerAgentCharacterData(const PacketContainer &packet, const pk2::media::ItemData &itemData, const pk2::media::SkillData &skillData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint32_t serverTime = stream.Read<uint32_t>();
  refObjId_ = stream.Read<uint32_t>();
  uint8_t scale = stream.Read<uint8_t>();
  uint8_t curLevel = stream.Read<uint8_t>();
  uint8_t maxLevel = stream.Read<uint8_t>();
  uint64_t expOffset = stream.Read<uint64_t>();
  uint32_t sExpOffset = stream.Read<uint32_t>();
  uint64_t remainGold = stream.Read<uint64_t>();
  uint32_t remainSkillPoint = stream.Read<uint32_t>();
  uint16_t remainStatPoint = stream.Read<uint16_t>();
  uint8_t remainHwanCount = stream.Read<uint8_t>();
  uint32_t gatheredExpPoint = stream.Read<uint32_t>();
  hp_ = stream.Read<uint32_t>();
  mp_ = stream.Read<uint32_t>();
  uint8_t autoInverstExp = stream.Read<uint8_t>();
  uint8_t dailyPK = stream.Read<uint8_t>();
  uint16_t totalPK = stream.Read<uint16_t>();
  uint32_t pKPenaltyPoint = stream.Read<uint32_t>();
  uint8_t hwanLevel = stream.Read<uint8_t>();
  uint8_t freePVP = stream.Read<uint8_t>(); // 0 = None, 1 = Red, 2 = Gray, 3 = Blue, 4 = White, 5 = Gold

  //=====================================================================================
  //===================================== Inventory =====================================
  //=====================================================================================

  inventorySize_ = stream.Read<uint8_t>();

  uint8_t inventoryItemCount = stream.Read<uint8_t>();

  for (int itemNum=0; itemNum<inventoryItemCount; ++itemNum) {
    uint8_t slotNum = stream.Read<uint8_t>();
    auto rentInfo = parseRentInfo(stream);

    uint32_t refItemId = stream.Read<uint32_t>();
    if (!itemData.haveItemWithId(refItemId)) {
      throw std::runtime_error("Unable to parse packet. Encountered an item (id:"+std::to_string(refItemId)+") for which we have no data on.");
    }
    const pk2::media::Item &item = itemData.getItemById(refItemId);

    item::Item *parsedItem = newItemByTypeData(item);
    if (parsedItem == nullptr) {
      throw std::runtime_error("Unable to create an item object for item");
    }

    parsedItem->refItemId = refItemId;
    parsedItem->itemInfo = &item;

    parseItem(parsedItem, stream);

    inventoryItemMap_.insert(std::pair<uint8_t, std::shared_ptr<item::Item>>(slotNum, parsedItem));
  }

  //=====================================================================================
  //================================== Avatar Inventory =================================
  //=====================================================================================

  uint8_t avatarInventorySize = stream.Read<uint8_t>();
  uint8_t avatarItemCount = stream.Read<uint8_t>();

  for (int i=0; i<avatarItemCount; ++i) {
    uint8_t slotNum = stream.Read<uint8_t>();
    uint32_t rentType = stream.Read<uint32_t>(); // TODO: Enum for this
    
    // TODO: Move the block below into a function? It is duplicate code
    if (rentType == 1) {
      uint16_t canDelete = stream.Read<uint16_t>();
      uint32_t periodBeginTime = stream.Read<uint32_t>();
      uint32_t periodEndTime = stream.Read<uint32_t>();
    } else if (rentType == 2) {
      uint16_t canDelete = stream.Read<uint16_t>();
      uint16_t canRecharge = stream.Read<uint16_t>();
      uint32_t meterRateTime = stream.Read<uint32_t>();
    } else if (rentType == 3) {
      uint16_t canDelete = stream.Read<uint16_t>();
      uint16_t canRecharge = stream.Read<uint16_t>();
      uint32_t periodBeginTime = stream.Read<uint32_t>();
      uint32_t periodEndTime = stream.Read<uint32_t>();
      uint32_t packingTime = stream.Read<uint32_t>();
    }

    uint32_t refItemId = stream.Read<uint32_t>();
    if (!itemData.haveItemWithId(refItemId)) {
      throw std::runtime_error("Unable to parse packet. Encountered an item (id:"+std::to_string(refItemId)+") for which we have no data on.");
    }
    const pk2::media::Item &item = itemData.getItemById(refItemId);
      
    if (item.typeId1 == 3) {
      // ITEM_
      if (item.typeId2 == 1) { //TODO: Narrow filters for AvatarInventory
        // ITEM_CH
        // ITEM_EU
        // AVATAR_
        item::ItemEquipment parsedItem;
        parseItem(parsedItem, stream);
      }
    }
  }

  uint8_t unknownByte0 = stream.Read<uint8_t>(); // "not a counter"

  //=====================================================================================
  //===================================== Masteries =====================================
  //=====================================================================================

  uint8_t hasNextMastery = stream.Read<uint8_t>();
  while (hasNextMastery == 1) {
    uint32_t id = stream.Read<uint32_t>();
    uint8_t level = stream.Read<uint8_t>();
    hasNextMastery = stream.Read<uint8_t>();
  }

  uint8_t unknownByte1 = stream.Read<uint8_t>(); // "not a counter"

  //=====================================================================================
  //====================================== Skills =======================================
  //=====================================================================================

  uint8_t hasNextSkill = stream.Read<uint8_t>();
  while (hasNextSkill == 1) {
    uint32_t id = stream.Read<uint32_t>();
    uint8_t enabled = stream.Read<uint8_t>();
    hasNextSkill = stream.Read<uint8_t>();
  }

  //=====================================================================================
  //====================================== Quests =======================================
  //=====================================================================================

  uint16_t completedQuestCount = stream.Read<uint16_t>();

  for (int i=0; i<completedQuestCount; ++i) {
    uint32_t quest = stream.Read<uint32_t>();
  }

  uint8_t activeQuestCount = stream.Read<uint8_t>();

  for (int i=0; i<activeQuestCount; ++i) {
    uint32_t refQuestID = stream.Read<uint32_t>();
    uint8_t achivementCount = stream.Read<uint8_t>();
    uint8_t requiresAutoShareParty = stream.Read<uint8_t>();
    uint8_t type = stream.Read<uint8_t>();

    if (type == 28) {
      uint32_t remainingTime = stream.Read<uint32_t>();
    }
    uint8_t questStatus = stream.Read<uint8_t>();
    
    if (type != 8) {
      uint8_t objectiveCount = stream.Read<uint8_t>();

      for (int j=0; j<objectiveCount; ++j) {
        uint8_t objectiveId = stream.Read<uint8_t>();
        uint8_t objectiveStatus = stream.Read<uint8_t>(); //0 = Done, 1  = On
        uint16_t objectiveNameLength = stream.Read<uint16_t>();
        std::string objectiveName = stream.Read_Ascii(objectiveNameLength);
        uint8_t objectiveTaskCount = stream.Read<uint8_t>();

        for (int k=0; k<objectiveTaskCount; ++k) {
          uint32_t value = stream.Read<uint32_t>();
        }
      }
    }

    if (type == 88) {
      uint8_t refObjectCount = stream.Read<uint8_t>();
      for (int j=0; j<refObjectCount; ++j) {
        uint32_t refObjID = stream.Read<uint32_t>();
      }
    }
  }

  uint8_t unknownByte2 = stream.Read<uint8_t>(); // "Structure changes!!!"

  //=====================================================================================
  //================================== Collection Book ==================================
  //=====================================================================================

  uint32_t collectionBookStartedThemeCount = stream.Read<uint32_t>();

  for (uint32_t i=0; i<collectionBookStartedThemeCount; ++i) {
    uint32_t themeIndex = stream.Read<uint32_t>();
    uint32_t themeStartedDateTime = stream.Read<uint32_t>(); // SROTimeStamp
    uint32_t themePages = stream.Read<uint32_t>();
  }
  
  entityUniqueId_ = stream.Read<uint32_t>();

  //=====================================================================================
  //===================================== Position ======================================
  //=====================================================================================

  // //Position
  uint16_t regionId = stream.Read<uint16_t>();
  float posX = stream.Read<float>();
  float posY = stream.Read<float>();
  float posZ = stream.Read<float>();
  uint16_t angle = stream.Read<uint16_t>();

  //=====================================================================================
  //===================================== Movement ======================================
  //=====================================================================================

  uint8_t hasDestination = stream.Read<uint8_t>();
  uint8_t movementType = stream.Read<uint8_t>();

  if (hasDestination) {
    uint16_t destinationRegion = stream.Read<uint16_t>();

    if (regionId < std::numeric_limits<uint16_t>::max()) {
      // World
      uint16_t destinationX = stream.Read<uint16_t>();
      uint16_t destinationY = stream.Read<uint16_t>();
      uint16_t destinationZ = stream.Read<uint16_t>();
    } else {
      // Dungeon
      uint32_t destinationOffsetX = stream.Read<uint32_t>();
      uint32_t destinationOffsetY = stream.Read<uint32_t>();
      uint32_t destinationOffsetZ = stream.Read<uint32_t>();
    }
  } else {
    uint8_t source = stream.Read<uint8_t>(); // 0 = Spinning, 1 = Sky-/Key-walking
    uint16_t angle = stream.Read<uint16_t>(); // Represents the new angle, character is looking at
  }

  //=====================================================================================
  //======================================= State =======================================
  //=====================================================================================

  uint8_t lifeState = stream.Read<uint8_t>(); // 1 = Alive, 2 = Dead
  uint8_t unkByte0 = stream.Read<uint8_t>();
  uint8_t motionState = stream.Read<uint8_t>(); // 0 = None, 2 = Walking, 3 = Running, 4 = Sitting
  uint8_t status = stream.Read<uint8_t>(); // 0 = None, 1 = Hwan, 2 = Untouchable, 3 = GameMasterInvincible, 5 = GameMasterInvisible, 5 = ?, 6 = Stealth, 7 = Invisible
  float walkSpeed = stream.Read<float>();
  float runSpeed = stream.Read<float>();
  float hwanSpeed = stream.Read<float>();
  uint8_t buffCount = stream.Read<uint8_t>();
  for (int i=0; i<buffCount; ++i) {
    uint32_t refSkillId = stream.Read<uint32_t>();
    uint32_t duration = stream.Read<uint32_t>();
    if (!skillData.haveSkillWithId(refSkillId)) {
      throw std::runtime_error("Unable to parse packet. Encountered an buff (id:"+std::to_string(refSkillId)+") for which we have no data on.");
    }
    const auto &skill = skillData.getSkillById(refSkillId);
    if (skill.isEfta()) {
      uint8_t creatorFlag = stream.Read<uint8_t>(); // 1=Creator, 2=Other
    }
  }

  uint16_t nameLength = stream.Read<uint16_t>();
  std::string name = stream.Read_Ascii(nameLength);
  uint16_t jobNameLength = stream.Read<uint16_t>();
  std::string jobName = stream.Read_Ascii(jobNameLength);
  uint8_t jobType = stream.Read<uint8_t>();
  uint8_t jobLevel = stream.Read<uint8_t>();
  uint32_t jobExp = stream.Read<uint32_t>();
  uint32_t jobContribution = stream.Read<uint32_t>();
  uint32_t jobReward = stream.Read<uint32_t>();
  uint8_t pvpState = stream.Read<uint8_t>(); // 0 = White, 1 = Purple, 2 = Red
  uint8_t transportFlag = stream.Read<uint8_t>();
  uint8_t inCombat = stream.Read<uint8_t>();

  if (transportFlag == 1) {
    uint32_t transportId = stream.Read<uint32_t>();
  }

  uint8_t pvpFlag = stream.Read<uint8_t>(); // 0 = Red Side, 1 = Blue Side, 0xFF = None
  uint64_t guideFlag = stream.Read<uint64_t>();
  uint32_t jId = stream.Read<uint32_t>();
  uint8_t gmFlag = stream.Read<uint8_t>();

  uint8_t activationFlag = stream.Read<uint8_t>(); // ConfigType:0 --> (0 = Not activated, 7 = activated)
  uint8_t hotkeyCount = stream.Read<uint8_t>(); // ConfigType:1
  
  for (int i=0; i<hotkeyCount; ++i) {
    uint8_t slotSeq = stream.Read<uint8_t>();
    uint8_t slotContentType = stream.Read<uint8_t>();
    uint32_t slotData = stream.Read<uint32_t>();
  }

  uint16_t autoHPConfig = stream.Read<uint16_t>(); // ConfigType:11
  uint16_t autoMPConfig = stream.Read<uint16_t>(); // ConfigType:12
  uint16_t autoUniversalConfig = stream.Read<uint16_t>(); // ConfigType:13
  uint8_t autoPotionDelay = stream.Read<uint8_t>(); // ConfigType:14

  uint8_t blockedWhisperCount = stream.Read<uint8_t>();

  for (int i=0; i<blockedWhisperCount; ++i) {
    uint16_t targetLength = stream.Read<uint16_t>();
    std::string target = stream.Read_Ascii(targetLength);
  }

  uint32_t unknownShort0 = stream.Read<uint32_t>(); //Structure changes!!!
  uint8_t unknownByte3 = stream.Read<uint8_t>(); //Structure changes!!!
}

//=========================================================================================================================================================

std::shared_ptr<Object> newObjectFromId(uint32_t refObjId, const pk2::media::CharacterData &characterData, const pk2::media::ItemData &itemData, const pk2::media::TeleportData &teleportData) {
  if (characterData.haveCharacterWithId(refObjId) &&
      characterData.getCharacterById(refObjId).typeId1 == 1) {
    const auto &character = characterData.getCharacterById(refObjId);
    if (character.typeId2 == 1) {
      auto ptr = std::make_shared<PlayerCharacter>();
      ptr->refObjId = refObjId;
      ptr->typeId1 = character.typeId1;
      ptr->typeId2 = character.typeId2;
      ptr->typeId3 = character.typeId3;
      ptr->typeId4 = character.typeId4;
      return ptr;
    } else if (character.typeId2 == 2 && character.typeId3 == 1) {
      auto ptr = std::make_shared<Monster>();
      ptr->refObjId = refObjId;
      ptr->typeId1 = character.typeId1;
      ptr->typeId2 = character.typeId2;
      ptr->typeId3 = character.typeId3;
      ptr->typeId4 = character.typeId4;
      return ptr;
    } else {
      auto ptr = std::make_shared<NonplayerCharacter>();
      ptr->refObjId = refObjId;
      ptr->typeId1 = character.typeId1;
      ptr->typeId2 = character.typeId2;
      ptr->typeId3 = character.typeId3;
      ptr->typeId4 = character.typeId4;
      return ptr;
    }
  } else if (itemData.haveItemWithId(refObjId) && itemData.getItemById(refObjId).typeId1 == 3) {
    const auto &item = itemData.getItemById(refObjId);
    auto ptr = std::make_shared<Item>();
    ptr->refObjId = refObjId;
    ptr->typeId1 = item.typeId1;
    ptr->typeId2 = item.typeId2;
    ptr->typeId3 = item.typeId3;
    ptr->typeId4 = item.typeId4;
    return ptr;
  } else if (teleportData.haveTeleportWithId(refObjId) && teleportData.getTeleportById(refObjId).typeId1 == 4) {
    const auto &portal = teleportData.getTeleportById(refObjId);
    auto ptr = std::make_shared<Portal>();
    ptr->refObjId = refObjId;
    ptr->typeId1 = portal.typeId1;
    ptr->typeId2 = portal.typeId2;
    ptr->typeId3 = portal.typeId3;
    ptr->typeId4 = portal.typeId4;
    return ptr;
  }
}

std::shared_ptr<Object> parseSpawn(StreamUtility &stream,
                                   const pk2::media::CharacterData &characterData,
                                   const pk2::media::ItemData &itemData,
                                   const pk2::media::SkillData &skillData,
                                   const pk2::media::TeleportData &teleportData) {
  const uint32_t refObjId = stream.Read<uint32_t>();
  std::shared_ptr<Object> obj = newObjectFromId(refObjId, characterData, itemData, teleportData);
  if (characterData.haveCharacterWithId(obj->refObjId) && characterData.getCharacterById(obj->refObjId).typeId1 == 1) {
    const auto &character = characterData.getCharacterById(obj->refObjId);
    Character *characterPtr = dynamic_cast<Character*>(obj.get());
    if (characterPtr == nullptr) {
      throw std::runtime_error("parseSpawn, have a character, but the obj pointer cannot be cast to a Character");
    }
    bool characterHasJobEquipmentInInventory = false; // TODO: Create better mechanism
    // BIONIC:
    //  CHARACTER
    //  NPC
    //   NPC_FORTRESS_STRUCT
    //   NPC_MOB
    //   NPC_COS
    //   NPC_FORTRESS_COS   
    if (character.typeId2 == 1) {
      // CHARACTER
      uint8_t scale = stream.Read<uint8_t>();
      uint8_t hwanLevel = stream.Read<uint8_t>();
      uint8_t pvpCape = stream.Read<uint8_t>();         //0 = None, 1 = Red, 2 = Gray, 3 = Blue, 4 = White, 5 = Orange
      uint8_t autoInverstExp = stream.Read<uint8_t>();  //1 = Beginner Icon, 2 = Helpful, 3 = Beginner & Helpful
      
      // Inventory
      uint8_t inventorySize = stream.Read<uint8_t>();
      uint8_t inventoryItemCount = stream.Read<uint8_t>();
      for (int i=0; i<inventoryItemCount; ++i) {
        uint32_t itemRefId = stream.Read<uint32_t>();
        if (!itemData.haveItemWithId(itemRefId)) {
          throw std::runtime_error("Parsing ServerAgentGroupSpawn, found item in character's inventory which we have no data on");
        }
        const auto &item = itemData.getItemById(itemRefId);
        if (item.typeId1 == 3 && item.typeId2 == 1) {
          uint8_t optLevel = stream.Read<uint8_t>();
        }
        if (item.typeId1 == 3 && item.typeId2 == 1 && item.typeId3 == 7 && item.typeId4 < 4) {
          // TypeId4 values:
          //  1,2,3 are normal job
          //  3 Santa
          //  4 Free PVP
          //  6,7 new job
          characterHasJobEquipmentInInventory = true;
        }
      }

      // AvatarInventory
      uint8_t avatarInventorySize = stream.Read<uint8_t>();
      uint8_t avatarInventoryItemCount = stream.Read<uint8_t>();
      for (int i=0; i<avatarInventoryItemCount; ++i) {
        uint32_t itemRefId = stream.Read<uint32_t>();
        if (!itemData.haveItemWithId(itemRefId)) {
          throw std::runtime_error("Parsing ServerAgentGroupSpawn, found item in character's avatar inventory which we have no data on");
        }
        const auto &item = itemData.getItemById(itemRefId);
        if (item.typeId1 == 3 && item.typeId2 == 1) {
          uint8_t optLevel = stream.Read<uint8_t>();
        }
      }

      // Mask
      bool hasMask = stream.Read<uint8_t>();
      if (hasMask) {
        uint32_t maskRefObjId = stream.Read<uint32_t>();
        if (!itemData.haveItemWithId(maskRefObjId)) {
          throw std::runtime_error("Parsing ServerAgentGroupSpawn, found mask on character which we have no data on");
        }
        const auto &maskItem = itemData.getItemById(maskRefObjId);
        if (maskItem.typeId1 == character.typeId1 &&
            maskItem.typeId2 == character.typeId2) {
          // Duplicate
          uint8_t maskScale = stream.Read<uint8_t>();
          uint8_t maskItemCount = stream.Read<uint8_t>();
          for (int i=0; i<maskItemCount; ++i) {
            uint32_t maskItemRefId = stream.Read<uint32_t>();
          }
        }
      }
    } else if(character.typeId2 == 2 && character.typeId3 == 5) {
      //NPC_FORTRESS_STRUCT
      uint32_t structureHp = stream.Read<uint32_t>();
      uint32_t structureRefEventStructId = stream.Read<uint32_t>();
      uint16_t structureState = stream.Read<uint16_t>();
    }

    characterPtr->gId = stream.Read<uint32_t>();

    // Position
    characterPtr->regionId = stream.Read<uint16_t>();
    uint32_t x = stream.Read<uint32_t>(); // really a float
    characterPtr->x = *reinterpret_cast<float*>(&x);
    uint32_t y = stream.Read<uint32_t>(); // really a float
    characterPtr->y = *reinterpret_cast<float*>(&y);
    uint32_t z = stream.Read<uint32_t>(); // really a float
    characterPtr->z = *reinterpret_cast<float*>(&z);
    uint16_t angle = stream.Read<uint16_t>();

    bool movementHasDestination = stream.Read<uint8_t>();
    uint8_t movementType = stream.Read<uint8_t>();
    if (movementHasDestination) {
      // Mouse destination
      uint16_t destinationRegionId = stream.Read<uint16_t>();
      if (destinationRegionId & 0x8000) {
        // Dungeon
        uint32_t destinationOffsetX = stream.Read<uint32_t>();
        uint32_t destinationOffsetY = stream.Read<uint32_t>();
        uint32_t destinationOffsetZ = stream.Read<uint32_t>();
      } else {
        // World
        uint16_t destinationOffsetX = stream.Read<uint16_t>();
        uint16_t destinationOffsetY = stream.Read<uint16_t>();
        uint16_t destinationOffsetZ = stream.Read<uint16_t>();
      }
    } else {
      uint8_t movementSource = stream.Read<uint8_t>(); // 0=spinning, 1=FC_GO_FORWARD
      uint16_t movementAngle = stream.Read<uint16_t>(); // Represents the new angle, character is looking at
    }
  
    // State
    uint8_t lifeState = stream.Read<uint8_t>(); // 1=Alive, 2=Dead
    uint8_t unkByte0 = stream.Read<uint8_t>(); // Obsolete
    uint8_t motionState = stream.Read<uint8_t>(); // 0=None, 2=Walking, 3=Running, 4=Sitting
    uint8_t bodyState = stream.Read<uint8_t>(); // 0=None, 1=Hwan, 2=Untouchable, 3=GameMasterInvincible, 5=GameMasterInvisible, 6=Stealth, 7=Invisible
    uint32_t walkSpeed = stream.Read<uint32_t>();
    uint32_t runSpeed = stream.Read<uint32_t>();
    uint32_t hwanSpeed = stream.Read<uint32_t>();

    // Buffs
    uint8_t buffCount = stream.Read<uint8_t>();
    for (int i=0; i<buffCount; ++i) {
      uint32_t skillRefId = stream.Read<uint32_t>();
      uint32_t token = stream.Read<uint32_t>();
      if (!skillData.haveSkillWithId(skillRefId)) {
        throw std::runtime_error("Parsing ServerAgentGroupSpawn, found buff which we have no data on"); // TODO: Add skill id to error
        // TODO: Also, I think we need to print this error when we catch the exception
      }
      const auto &skill = skillData.getSkillById(skillRefId);
      if (skill.isEfta()) {
        uint8_t creatorFlag = stream.Read<uint8_t>(); // 1=Creator, 2=Other
      }
    }

    if (character.typeId2 == 1) {
      // CHARACTER
      PlayerCharacter *playerCharacterPtr = dynamic_cast<PlayerCharacter*>(obj.get());
      if (playerCharacterPtr == nullptr) {
        throw std::runtime_error("parseSpawn, have a player character, but the obj pointer cannot be cast to a PlayerCharacter");
      }
      uint16_t nameLength = stream.Read<uint16_t>();
      playerCharacterPtr->name = stream.Read_Ascii(nameLength);
      
      uint8_t jobType = stream.Read<uint8_t>(); // 0=None, 1=Trader, 2=Tief, 3=Hunter
      uint8_t jobLevel = stream.Read<uint8_t>();
      uint8_t murderState = stream.Read<uint8_t>(); //0=White (Neutral), 1=Purple (Assaulter), 2=Red (Murder)
      bool isRiding = stream.Read<uint8_t>();
      bool inCombat = stream.Read<uint8_t>();
      if(isRiding) {
        uint32_t transportUniqueId = stream.Read<uint32_t>();
      }
      uint8_t scrollMode = stream.Read<uint8_t>(); // 0=None, 1=Return Scroll, 2=Bandit Return Scroll
      uint8_t interactMode = stream.Read<uint8_t>(); //0=None 2=P2P, 4=P2N_TALK, 6=OPNMKT_DEAL
      // exchange, talking to npc, stalling, etc
      uint8_t unkByte4 = stream.Read<uint8_t>();

      //Guild
      uint16_t guildNameLength = stream.Read<uint16_t>();
      std::string guildName = stream.Read_Ascii(guildNameLength);
      
      if (!characterHasJobEquipmentInInventory) {
        uint32_t guildId = stream.Read<uint32_t>();
        uint16_t guildMemberNicknameLength = stream.Read<uint16_t>();
        std::string guildMemberNickname = stream.Read_Ascii(guildMemberNicknameLength);
        uint32_t guildLastCrestRev = stream.Read<uint32_t>();
        uint32_t unionId = stream.Read<uint32_t>();
        uint32_t unionLastCrestRev = stream.Read<uint32_t>();
        uint8_t guildIsFriendly = stream.Read<uint8_t>(); // 0 = Hostile, 1 = Friendly
        uint8_t guildMemberSiegeAuthority = stream.Read<uint8_t>(); // See SiegeAuthority.cs
      }

      if (interactMode == 4) {
        uint16_t stallNameLength = stream.Read<uint16_t>();
        std::string stallName = stream.Read_Ascii(stallNameLength);
        uint32_t stallDecorationRefId = stream.Read<uint32_t>();
      }

      uint8_t equipmentCooldown = stream.Read<uint8_t>(); // Yellow bar when equipping or unequipping
      uint8_t pkFlag = stream.Read<uint8_t>();
    } else if (character.typeId2 == 2) {
      // NPC
      uint8_t talkFlag = stream.Read<uint8_t>();
      if (talkFlag == 2) {
        uint8_t talkOptionCount = stream.Read<uint8_t>();
        std::vector<uint8_t> talkOptions;
        for (int i=0; i<talkOptionCount; ++i) {
          talkOptions.emplace_back(stream.Read<uint8_t>());
        }
      }
      
      if (character.typeId3 == 1) {
        // NPC_MOB
        Monster *monsterPtr = dynamic_cast<Monster*>(obj.get());
        if (monsterPtr == nullptr) {
          throw std::runtime_error("parseSpawn, have a monster, but the obj pointer cannot be cast to a Monster");
        }
        monsterPtr->monsterRarity = stream.Read<uint8_t>();
        if (character.typeId4 == 2 || character.typeId4 == 3) {
          // NPC_MOB_THIEF, NPC_MOB_HUNTER
          uint8_t appearance = stream.Read<uint8_t>();
        }
      } else if (character.typeId3 == 3) {
        // NPC_COS
        if (character.typeId4 == 2 || // NPC_COS_TRANSPORT
            character.typeId4 == 3 || // NPC_COS_P_GROWTH
            character.typeId4 == 4 || // NPC_COS_P_ABILITY
            character.typeId4 == 5 || // NPC_COS_GUILD
            character.typeId4 == 6 || // NPC_COS_CAPTURED
            character.typeId4 == 7 || // NPC_COS_QUEST
            character.typeId4 == 8) { // NPC_COS_QUEST
          if (character.typeId4 == 3 || character.typeId4 == 4) { // NPC_COS_P_GROWTH, NPC_COS_P_ABILITY
            uint16_t nameLength = stream.Read<uint16_t>();
            std::string name = stream.Read_Ascii(nameLength);
          } else if (character.typeId4 == 5) { // NPC_COS_GUILD
            uint16_t guildNameLength = stream.Read<uint16_t>();
            std::string guildName = stream.Read_Ascii(guildNameLength);
          }
          
          if (character.typeId4 == 2 || // NPC_COS_TRANSPORT
              character.typeId4 == 3 || // NPC_COS_P_GROWTH
              character.typeId4 == 4 || // NPC_COS_P_ABILITY
              character.typeId4 == 5 || // NPC_COS_GUILD
              character.typeId4 == 6) { // NPC_COS_CAPTURED
            uint16_t ownerNameLength = stream.Read<uint16_t>();
            std::string ownerName = stream.Read_Ascii(ownerNameLength);
                  
            if (character.typeId4 == 2 || // NPC_COS_TRANSPORT
                character.typeId4 == 3 || // NPC_COS_P_GROWTH
                character.typeId4 == 4 || // NPC_COS_P_ABILITY
                character.typeId4 == 5) { // NPC_COS_GUILD
              uint8_t ownerJobType = stream.Read<uint8_t>();
                          
              if (character.typeId4 == 2 || // NPC_COS_TRANSPORT
                  character.typeId4 == 3 || // NPC_COS_P_GROWTH
                  character.typeId4 == 5) { // NPC_COS_GUILD
                uint8_t ownerPvpState = stream.Read<uint8_t>();
                if (character.typeId4 == 5) { //NPC_COS_GUILD
                  uint32_t ownerRefId = stream.Read<uint32_t>();
                }
              }
            }
          }
          uint32_t ownerUniqueId = stream.Read<uint32_t>();
        }
      } else if (character.typeId3 == 4) {
        // GObjSiegeObject
        // NPC_FORTRESS_COS
        uint32_t guildId = stream.Read<uint32_t>();
        uint16_t guildNameLength = stream.Read<uint16_t>();
        std::string guildName = stream.Read_Ascii(guildNameLength);
      }
    }
  } else if (itemData.haveItemWithId(obj->refObjId) && itemData.getItemById(obj->refObjId).typeId1 == 3) {
    Item *itemPtr = dynamic_cast<Item*>(obj.get());
    if (itemPtr == nullptr) {
      throw std::runtime_error("parseSpawn, have an item, but the obj pointer cannot be cast to a Item");
    }
    const auto &item = itemData.getItemById(obj->refObjId);
    // ITEM
    //  ITEM_EQUIP
    //  ITEM_ETC
    //   ITEM_ETC_MONEY_GOLD
    //   ITEM_ETC_TRADE
    //   ITEM_ETC_QUEST
    if (item.typeId2 == 1) {
      // ITEM_EQUIP
      uint8_t optLevel = stream.Read<uint8_t>();
    } else if (item.typeId2 == 3) {
      // ITEM_ETC
      if (item.typeId3 == 5 && item.typeId4 == 0) {
        // ITEM_ETC_MONEY_GOLD
        uint32_t goldAmount = stream.Read<uint32_t>();
      } else if (item.typeId3 == 8 || item.typeId4 == 9) {
        // ITEM_ETC_TRADE
        // ITEM_ETC_QUEST
        uint16_t ownerNameLength = stream.Read<uint16_t>();
        std::string ownerName = stream.Read_Ascii(ownerNameLength);
      }
    }
    itemPtr->gId = stream.Read<uint32_t>();
    itemPtr->regionId = stream.Read<uint16_t>();
    uint32_t x = stream.Read<uint32_t>(); // Actually a float
    itemPtr->x = *reinterpret_cast<float*>(&x);
    uint32_t y = stream.Read<uint32_t>(); // Actually a float
    itemPtr->y = *reinterpret_cast<float*>(&y);
    uint32_t z = stream.Read<uint32_t>(); // Actually a float
    itemPtr->z = *reinterpret_cast<float*>(&z);
    uint16_t angle = stream.Read<uint16_t>();
    bool hasOwner = stream.Read<uint8_t>();
    if (hasOwner) {
      uint32_t ownerJId = stream.Read<uint32_t>();
    }
    itemPtr->rarity = stream.Read<uint8_t>(); // Educated guess: 0=white, 1=blue, 2=sox
  } else if (teleportData.haveTeleportWithId(obj->refObjId) && teleportData.getTeleportById(obj->refObjId).typeId1 == 4) {
    Portal *portalPtr = dynamic_cast<Portal*>(obj.get());
    if (portalPtr == nullptr) {
      throw std::runtime_error("parseSpawn, have a portal, but the obj pointer cannot be cast to a Portal");
    }
    // PORTALS
    //  STORE
    //  INS_TELEPORTER
    portalPtr->gId = stream.Read<uint32_t>();
    portalPtr->regionId = stream.Read<uint16_t>();
    uint32_t x = stream.Read<uint32_t>(); // Actually a float
    portalPtr->x = *reinterpret_cast<float*>(&x);
    uint32_t y = stream.Read<uint32_t>(); // Actually a float
    portalPtr->y = *reinterpret_cast<float*>(&y);
    uint32_t z = stream.Read<uint32_t>(); // Actually a float
    portalPtr->z = *reinterpret_cast<float*>(&z);
    uint16_t angle = stream.Read<uint16_t>();

    uint8_t unkByte0 = stream.Read<uint8_t>();
    uint8_t unkByte1 = stream.Read<uint8_t>();
    uint8_t unkByte2 = stream.Read<uint8_t>();
    portalPtr->unkByte3 = stream.Read<uint8_t>();

    if (portalPtr->unkByte3 == 1) {
      // Regular
      uint32_t unkUInt0 = stream.Read<uint32_t>();
      uint32_t unkUInt1 = stream.Read<uint32_t>();
    } else if (portalPtr->unkByte3 == 6) {
      // Dimension Hole
      uint16_t ownerNameLength = stream.Read<uint16_t>();
      std::string ownerName = stream.Read_Ascii(ownerNameLength);
      uint32_t ownerUId = stream.Read<uint32_t>();
    }

    if (unkByte1 == 1) {
      // STORE_EVENTZONE_DEFAULT
      uint32_t unkUint2 = stream.Read<uint32_t>();
      uint8_t unkByte4 = stream.Read<uint8_t>();
    }
  } else if (obj->refObjId == std::numeric_limits<uint32_t>::max()) {
    std::cout << "EVENT_ZONE\n";
    // EVENT_ZONE (Traps, Buffzones, ...)
    uint16_t eventZoneTypeId = stream.Read<uint16_t>();
    std::cout << " eventZoneTypeId:" << eventZoneTypeId << '\n';
    uint32_t eventZoneRefSkillId = stream.Read<uint32_t>();
    std::cout << " eventZoneRefSkillId:" << eventZoneRefSkillId << '\n';
    uint32_t uniqueId = stream.Read<uint32_t>();
    std::cout << " uniqueId:" << uniqueId << '\n';
    uint16_t regionId = stream.Read<uint16_t>();
    std::cout << " regionId:" << regionId << '\n';
    uint32_t x = stream.Read<uint32_t>(); // Actually a float
    std::cout << " x:" << *reinterpret_cast<float*>(&x) << '\n';
    uint32_t y = stream.Read<uint32_t>(); // Actually a float
    std::cout << " y:" << *reinterpret_cast<float*>(&y) << '\n';
    uint32_t z = stream.Read<uint32_t>(); // Actually a float
    std::cout << " z:" << *reinterpret_cast<float*>(&z) << '\n';
    uint16_t angle = stream.Read<uint16_t>();
    std::cout << " angle:" << angle << '\n';
  }
  return obj;
}

uint32_t parseDespawn(StreamUtility &stream) {
  return stream.Read<uint32_t>();
}

ParsedServerAgentGroupSpawn::ParsedServerAgentGroupSpawn(const PacketContainer &packet,
                                                         const pk2::media::CharacterData &characterData,
                                                         const pk2::media::ItemData &itemData,
                                                         const pk2::media::SkillData &skillData,
                                                         const pk2::media::TeleportData &teleportData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  // This data is originally from the begin packet (before the data packet)
  groupSpawnType_ = static_cast<GroupSpawnType>(stream.Read<uint8_t>());
  uint16_t groupSpawnAmount = stream.Read<uint16_t>();
  if (groupSpawnType_ == GroupSpawnType::kSpawn) {
    for (int spawnNum=0; spawnNum<groupSpawnAmount; ++spawnNum) {
      objects_.emplace_back(parseSpawn(stream, characterData, itemData, skillData, teleportData));
    }
  } else if (groupSpawnType_ == GroupSpawnType::kDespawn) {
    for (int despawnNum=0; despawnNum<groupSpawnAmount; ++despawnNum) {
      despawns_.emplace_back(parseDespawn(stream));
    }
  }
}

GroupSpawnType ParsedServerAgentGroupSpawn::groupSpawnType() const {
  return groupSpawnType_;
}

const std::vector<std::shared_ptr<Object>>& ParsedServerAgentGroupSpawn::objects() const {
  return objects_;
}

const std::vector<uint32_t>& ParsedServerAgentGroupSpawn::despawns() const {
  return despawns_;
}

//=========================================================================================================================================================

ParsedServerAgentSpawn::ParsedServerAgentSpawn(const PacketContainer &packet,
                                               const pk2::media::CharacterData &characterData,
                                               const pk2::media::ItemData &itemData,
                                               const pk2::media::SkillData &skillData,
                                               const pk2::media::TeleportData &teleportData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  object_ = parseSpawn(stream, characterData, itemData, skillData, teleportData);
  if (object_->typeId1 == 1 || object_->typeId1 == 4) {
    //BIONIC and STORE
    uint8_t spawnType = stream.Read<uint8_t>(); // 1=COS_SUMMON, 3=SPAWN, 4=SPAWN_WALK
  } else if (object_->typeId1 == 3) {
    uint8_t dropSource = stream.Read<uint8_t>();
    uint32_t dropperUniqueId = stream.Read<uint32_t>();
  }
}

std::shared_ptr<Object> ParsedServerAgentSpawn::object() const {
  return object_;
}
                         
//=========================================================================================================================================================

ParsedServerAgentDespawn::ParsedServerAgentDespawn(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  gId_ = parseDespawn(stream);
}

uint32_t ParsedServerAgentDespawn::gId() const {
  return gId_;
}

//=========================================================================================================================================================

ParsedServerAgentCharacterSelectionJoinResponse::ParsedServerAgentCharacterSelectionJoinResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = stream.Read<uint8_t>();
  if (result_ == 0x02) {
    errorCode_ = stream.Read<uint16_t>();
  }
}

uint8_t ParsedServerAgentCharacterSelectionJoinResponse::result() const {
  return result_;
}

uint16_t ParsedServerAgentCharacterSelectionJoinResponse::errorCode() const {
  return errorCode_;
}

//=========================================================================================================================================================

ParsedServerAgentCharacterSelectionActionResponse::ParsedServerAgentCharacterSelectionActionResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  action_ = static_cast<packet_enums::CharacterSelectionAction>(stream.Read<uint8_t>());
  result_ = stream.Read<uint8_t>();
  if (result_ == 0x01 && action_ == packet_enums::CharacterSelectionAction::kList) {
    // Listing characters
    const uint8_t kCharCount = stream.Read<uint8_t>();
    for (int i=0; i<kCharCount; ++i) {
      PacketInnerStructures::CharacterSelection::Character character;
      character.refObjID = stream.Read<uint32_t>();
      const uint16_t kNameLength = stream.Read<uint16_t>();
      character.name = stream.Read_Ascii(kNameLength);
      character.scale = stream.Read<uint8_t>();
      character.curLevel = stream.Read<uint8_t>();
      character.expOffset = stream.Read<uint64_t>();
      character.strength = stream.Read<uint16_t>();
      character.intelligence = stream.Read<uint16_t>();
      character.statPoint = stream.Read<uint16_t>();
      character.curHP = stream.Read<uint32_t>();
      character.curMP = stream.Read<uint32_t>();
      character.isDeleting = stream.Read<bool>();
      if (character.isDeleting) {
        character.charDeleteTime = stream.Read<uint32_t>();
      }
      character.guildMemberClass = stream.Read<uint8_t>();
      character.isGuildRenameRequired = stream.Read<bool>();
      if (character.isGuildRenameRequired) {
        const uint16_t kCurrentGuildNameLength = stream.Read<uint16_t>();
        character.currentGuildName = stream.Read_Ascii(kCurrentGuildNameLength);
      }
      character.academyMemberClass = stream.Read<uint8_t>();
      const uint8_t kItemCount = stream.Read<uint8_t>();
      for (int j=0; j<kItemCount; ++j) {
        PacketInnerStructures::CharacterSelection::Item item;   
        item.refId = stream.Read<uint32_t>();
        item.plus = stream.Read<uint8_t>();
        character.items.emplace_back(std::move(item));
      }
      const uint8_t kAvatarCount = stream.Read<uint8_t>();
      for (int j=0; j<kAvatarCount; ++j) {
        PacketInnerStructures::CharacterSelection::Avatar avatar;   
        avatar.refId = stream.Read<uint32_t>();
        avatar.plus = stream.Read<uint8_t>();
        character.avatars.emplace_back(std::move(avatar));
      }
      characters_.emplace_back(std::move(character));
    }
  } else if (result_ == 0x02) {
    errorCode_ = stream.Read<uint16_t>();
  }
}

packet_enums::CharacterSelectionAction ParsedServerAgentCharacterSelectionActionResponse::action() const {
  return action_;
}

uint8_t ParsedServerAgentCharacterSelectionActionResponse::result() const {
  return result_;
}

const std::vector<PacketInnerStructures::CharacterSelection::Character>& ParsedServerAgentCharacterSelectionActionResponse::characters() const {
  return characters_;
}

uint16_t ParsedServerAgentCharacterSelectionActionResponse::errorCode() const {
  return errorCode_;
}

//=========================================================================================================================================================

ParsedServerAuthResponse::ParsedServerAuthResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = stream.Read<uint8_t>();
  if (result_ == 0x02) {
    errorCode_ = stream.Read<uint8_t>();
  }
}

uint8_t ParsedServerAuthResponse::result() const {
  return result_;
}

uint8_t ParsedServerAuthResponse::errorCode() const {
  return errorCode_;
}

//=========================================================================================================================================================

ParsedLoginClientInfo::ParsedLoginClientInfo(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint16_t serviceNameLength = stream.Read<uint16_t>();
  serviceName_ = stream.Read_Ascii(serviceNameLength);
  /* uint8_t isLocal = */ stream.Read<uint8_t>();
}

std::string ParsedLoginClientInfo::serviceName() const {
  return serviceName_;
}

//=========================================================================================================================================================

ParsedLoginResponse::ParsedLoginResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = static_cast<packet_enums::LoginResult>(stream.Read<uint8_t>());
  if (result_ == packet_enums::LoginResult::kSuccess) {
    token_ = stream.Read<uint32_t>();
    uint16_t ipLength = stream.Read<uint16_t>();
    std::string ip = stream.Read_Ascii(ipLength);
    uint16_t port = stream.Read<uint16_t>();
  } else if (result_ == packet_enums::LoginResult::kFailed) {
    uint8_t errorCode = stream.Read<uint8_t>();
    if (errorCode == 0x01) {
      uint32_t maxAttempts = stream.Read<uint32_t>();
      uint32_t currentAttempts = stream.Read<uint32_t>();
    } else if (errorCode == 0x02) {
      packet_enums::LoginBlockType blockType = static_cast<packet_enums::LoginBlockType>(stream.Read<uint8_t>());
      if (blockType == packet_enums::LoginBlockType::kPunishment) {
        uint16_t reasonLength = stream.Read<uint16_t>();
        std::string reason = stream.Read_Ascii(reasonLength);
        uint16_t endDateYear = stream.Read<uint16_t>();
        uint16_t endDateMonth = stream.Read<uint16_t>();
        uint16_t endDateDay = stream.Read<uint16_t>();
        uint16_t endDateHour = stream.Read<uint16_t>();
        uint16_t endDateMinute = stream.Read<uint16_t>();
        uint16_t endDateSecond = stream.Read<uint16_t>();
        uint16_t endDateMicrosecond = stream.Read<uint16_t>();
      }
    }
  } else if (result_ == packet_enums::LoginResult::kOther) {
    /* uint8_t unkByte0 = */ stream.Read<uint8_t>();
    /* uint8_t unkByte1 = */ stream.Read<uint8_t>();
    uint16_t messageLength = stream.Read<uint16_t>();
    /* std::string message = */ stream.Read_Ascii(messageLength);
    /* uint16_t unkUShort0 = */ stream.Read<uint16_t>();
  }
}

packet_enums::LoginResult ParsedLoginResponse::result() const {
  return result_;
}

uint32_t ParsedLoginResponse::token() const {
  return token_;
}


//=========================================================================================================================================================

ParsedLoginServerList::ParsedLoginServerList(const PacketContainer &packet) :
    ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint8_t globalOpFlag = stream.Read<uint8_t>();
  while (globalOpFlag == 0x01) {
    // Read a "global op" , will be something like "SRO_Vietnam_TestLocal"
    uint8_t globalOpType = stream.Read<uint8_t>(); // For Atomix, its SRO_Taiwan_TestIn
    uint16_t globalNameLength = stream.Read<uint16_t>();
    std::string globalName = stream.Read_Ascii(globalNameLength);
    globalOpFlag = stream.Read<uint8_t>();
  }
  uint8_t shardFlag = stream.Read<uint8_t>();
  while (shardFlag == 0x01) {
    // Read a "shard" , will be something like "Atomix"
    shardId_ = stream.Read<uint16_t>();
    uint16_t shardNameLength = stream.Read<uint16_t>();
    std::string shardName = stream.Read_Ascii(shardNameLength);
    uint16_t shardCurrent = stream.Read<uint16_t>();
    uint16_t shardCapacity = stream.Read<uint16_t>();
    bool shardOnline = stream.Read<uint8_t>();
    uint8_t globalOpId = stream.Read<uint8_t>(); // Idk what this is, i guess globalOpType from above
    shardFlag = stream.Read<uint8_t>();
  }
}

uint16_t ParsedLoginServerList::shardId() const {
  return shardId_;
}

//=========================================================================================================================================================

ParsedClientChat::ParsedClientChat(const PacketContainer &packet) :
      ParsedPacket(packet) {
  // 1   byte    chatType
  // 1   byte    chatIndex
  // if(chatType == ChatType.PM)
  // {
  //     2   ushort  reciver.Length
  //     *   string  reciver
  // }
  // 2   ushort  message.Length
  // *   string  message
  StreamUtility stream = packet.data;
  chatType_ = static_cast<packet_enums::ChatType>(stream.Read<uint8_t>());
  chatIndex_ = stream.Read<uint8_t>();
  if (chatType_ == packet_enums::ChatType::kPm) {
    const uint16_t kReceiverNameLength = stream.Read<uint16_t>();
    receiverName_ = stream.Read_Ascii(kReceiverNameLength);
  }
  const uint16_t kMessageLength = stream.Read<uint16_t>();
  message_ = stream.Read_Ascii(kMessageLength);
}

packet_enums::ChatType ParsedClientChat::chatType() const {
  return chatType_;
}

uint8_t ParsedClientChat::chatIndex() const {
  return chatIndex_;
}

const std::string& ParsedClientChat::receiverName() const {
  return receiverName_;
}

const std::string& ParsedClientChat::message() const {
  return message_;
}

//=========================================================================================================================================================

ParsedClientItemMove::ParsedClientItemMove(const PacketContainer &packet) : ParsedPacket(packet) {
  std::cout << "ParsedClientItemMove\n";
  StreamUtility stream = packet.data;
  movement_.type = static_cast<packet_enums::ItemMovementType>(stream.Read<uint8_t>());
  if (movement_.type == packet_enums::ItemMovementType::kWithinInventory) {
    std::cout << "kWithinInventory\n";
    uint8_t sourceSlot = stream.Read<uint8_t>();
    std::cout << "sourceSlot: " << (int)sourceSlot << '\n';
    uint8_t destSlot = stream.Read<uint8_t>();
    std::cout << "destSlot: " << (int)destSlot << '\n';
    uint16_t quantity = stream.Read<uint16_t>();
    std::cout << "quantity: " << quantity << '\n';
  } else if (movement_.type == packet_enums::ItemMovementType::kWithinStorage ||
             movement_.type == packet_enums::ItemMovementType::kWithinGuildStorage) {
    std::cout << "kWithinStorage, kWithinGuildStorage\n";
    uint8_t sourceSlot = stream.Read<uint8_t>();
    std::cout << "sourceSlot: " << (int)sourceSlot << '\n';
    uint8_t destSlot = stream.Read<uint8_t>();
    std::cout << "destSlot: " << (int)destSlot << '\n';
    uint16_t quantity = stream.Read<uint16_t>();
    std::cout << "quantity: " << quantity << '\n';
    uint32_t unk0 = stream.Read<uint32_t>();
    std::cout << "unk0: " << unk0 << '\n';
  } else if (movement_.type == packet_enums::ItemMovementType::kInventoryToStorage ||
             movement_.type == packet_enums::ItemMovementType::kStorageToInventory ||
             movement_.type == packet_enums::ItemMovementType::kInventoryToGuildStorage ||
             movement_.type == packet_enums::ItemMovementType::kGuildStorageToInventory) {
    std::cout << "kInventoryToStorage, kStorageToInventory, kInventoryToGuildStorage, kGuildStorageToInventory\n";
    uint8_t sourceSlot = stream.Read<uint8_t>();
    std::cout << "sourceSlot: " << (int)sourceSlot << '\n';
    uint8_t destSlot = stream.Read<uint8_t>();
    std::cout << "destSlot: " << (int)destSlot << '\n';
    uint32_t unk0 = stream.Read<uint32_t>();
    std::cout << "unk0: " << unk0 << '\n';
  } else if (movement_.type == packet_enums::ItemMovementType::kBuyFromNPC) {
    std::cout << "kBuyFromNPC\n";
    uint8_t storeTab = stream.Read<uint8_t>();
    std::cout << "storeTab: " << (int)storeTab << '\n';
    uint8_t storeSlot = stream.Read<uint8_t>();
    std::cout << "storeSlot: " << (int)storeSlot << '\n';
    uint16_t quantity = stream.Read<uint16_t>();
    std::cout << "quantity: " << quantity << '\n';
    movement_.globalId = stream.Read<uint32_t>();
    std::cout << "globalId: " << movement_.globalId << '\n';
  } else if (movement_.type == packet_enums::ItemMovementType::kSellToNPC) {
    std::cout << "kSellToNPC\n";
    uint8_t sourceSlot = stream.Read<uint8_t>();
    std::cout << "sourceSlot: " << (int)sourceSlot << '\n';
    uint16_t quantity = stream.Read<uint16_t>();
    std::cout << "quantity: " << quantity << '\n';
    uint32_t unk1 = stream.Read<uint32_t>();
    std::cout << "unk1: " << unk1 << '\n';
  } else if (movement_.type == packet_enums::ItemMovementType::kGoldDrop ||
             movement_.type == packet_enums::ItemMovementType::kGoldStorageWithdraw ||
             movement_.type == packet_enums::ItemMovementType::kGoldStorageDeposit ||
             movement_.type == packet_enums::ItemMovementType::kGoldGuildStorageDeposit ||
             movement_.type == packet_enums::ItemMovementType::kGoldGuildStorageWithdraw) {
    std::cout << "kGoldDrop, kGoldStorageWithdraw, kGoldStorageDeposit, kGoldGuildStorageDeposit, kGoldGuildStorageWithdraw\n";
    uint64_t goldAmount = stream.Read<uint64_t>();
    std::cout << "goldAmount: " << goldAmount << '\n';
  } else if (movement_.type == packet_enums::ItemMovementType::kCosToInventory ||
             movement_.type == packet_enums::ItemMovementType::kInventoryToCos) {
    std::cout << "kCosToInventory,, kInventoryToCos\n";
    uint32_t unk4 = stream.Read<uint32_t>();
    std::cout << "unk4: " << unk4 << '\n';
    uint8_t sourceSlot = stream.Read<uint8_t>();
    std::cout << "sourceSlot: " << (int)sourceSlot << '\n';
    uint8_t destSlot = stream.Read<uint8_t>();
    std::cout << "destSlot: " << (int)destSlot << '\n';
  } else if (movement_.type == packet_enums::ItemMovementType::kWithinCos) {
    std::cout << "kWithinCos\n";
    uint32_t unk4 = stream.Read<uint32_t>();
    std::cout << "unk4: " << unk4 << '\n';
    uint8_t sourceSlot = stream.Read<uint8_t>();
    std::cout << "sourceSlot: " << (int)sourceSlot << '\n';
    uint8_t destSlot = stream.Read<uint8_t>();
    std::cout << "destSlot: " << (int)destSlot << '\n';
    uint16_t quantity = stream.Read<uint16_t>();
    std::cout << "quantity: " << quantity << '\n';
  } else if (movement_.type == packet_enums::ItemMovementType::kAvatarToInventory) {
    std::cout << "kAvatarToInventory\n";
    uint8_t sourceAvatarInventorySlot = stream.Read<uint8_t>();
    std::cout << "sourceAvatarInventorySlot: " << (int)sourceAvatarInventorySlot << '\n';
    uint8_t destInventorySlot = stream.Read<uint8_t>();
    std::cout << "destInventorySlot: " << (int)destInventorySlot << '\n';
  } else if (movement_.type == packet_enums::ItemMovementType::kInventoryToAvatar) {
    std::cout << "kInventoryToAvatar\n";
    uint8_t sourceInventorySlot = stream.Read<uint8_t>();
    std::cout << "sourceInventorySlot: " << (int)sourceInventorySlot << '\n';
    uint8_t destAvatarInventorySlot = stream.Read<uint8_t>();
    std::cout << "destAvatarInventorySlot: " << (int)destAvatarInventorySlot << '\n';
  } else {
    std::cout << "New item movement type! " << static_cast<int>(movement_.type) << '\n';
    std::cout << "Dump: " << DumpToString(stream) << '\n';
  }
}

ItemMovement ParsedClientItemMove::movement() const {
  return movement_;
}

//=========================================================================================================================================================
} // namespace packet::parsing