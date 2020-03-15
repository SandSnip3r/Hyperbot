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

uint32_t ParsedServerAgentCharacterData::entityUniqueId() const {
  return entityUniqueId_;
}

uint32_t ParsedServerAgentCharacterData::hp() const {
  return hp_;
}

uint32_t ParsedServerAgentCharacterData::mp() const {
  return mp_;
}

const std::map<uint8_t,ParsedServerAgentCharacterData::ItemVariantType>& ParsedServerAgentCharacterData::inventoryItemMap() const {
  return inventoryItemMap_;
}

item::ItemEquipment parseItemEquipment(StreamUtility &stream) {
  std::cout << "Parsing Item Equipment\n";
  item::ItemEquipment item;
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
  return item;
}

void parseItemCosSummoner(StreamUtility &stream, item::ItemCosGrowthSummoner *cosSummoner) {  
  cosSummoner->lifeState = static_cast<item::CosLifeState>(stream.Read<uint8_t>());
  if (cosSummoner->lifeState != item::CosLifeState::kInactive) {
    cosSummoner->refObjID = stream.Read<uint32_t>();
    uint16_t nameLength = stream.Read<uint16_t>();
    cosSummoner->name = stream.Read_Ascii(nameLength);

    // Special case for ability pets
    item::ItemCosAbilitySummoner *cosAbilitySummoner;
    if (cosAbilitySummoner = dynamic_cast<item::ItemCosAbilitySummoner*>(cosSummoner)) {
      cosAbilitySummoner->secondsToRentEndTime = stream.Read<uint32_t>();
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

item::ItemCosGrowthSummoner parseItemCosGrowthSummoner(StreamUtility &stream) {
  item::ItemCosGrowthSummoner item;
  parseItemCosSummoner(stream, &item);
  return item;
}

item::ItemCosAbilitySummoner parseItemCosAbilitySummoner(StreamUtility &stream) {
  item::ItemCosAbilitySummoner item;
  parseItemCosSummoner(stream, &item);
  return item;
}

item::ItemMonsterCapsule parseItemMonsterCapsule(StreamUtility &stream) {
  item::ItemMonsterCapsule item;
  item.refObjID = stream.Read<uint32_t>();
  return item;
}

item::ItemStorage parseItemStorage(StreamUtility &stream) {
  item::ItemStorage item;
  item.quantity = stream.Read<uint32_t>();
  return item;
}

item::ItemExpendable parseItemExpendable(StreamUtility &stream) {
  item::ItemExpendable item;
  item.stackCount = stream.Read<uint16_t>();
  return item;
}

item::ItemStone parseItemStone(StreamUtility &stream) {
  item::ItemStone item;
  item.stackCount = stream.Read<uint16_t>();
  item.attributeAssimilationProbability = stream.Read<uint8_t>();
  return item;
}

item::ItemMagicPop parseItemMagicPop(StreamUtility &stream) {
  item::ItemMagicPop item;
  item.stackCount = stream.Read<uint16_t>();
  uint8_t magParamCount = stream.Read<uint8_t>();
  for (int paramIndex=0; paramIndex<magParamCount; ++paramIndex) {
    item.magicParams.emplace_back();
    auto &magicParam = item.magicParams.back();
    magicParam.type = stream.Read<uint32_t>();
    magicParam.value = stream.Read<uint32_t>();
  }
  return item;
}

ParsedServerAgentCharacterData::ParsedServerAgentCharacterData(const PacketContainer &packet, const pk2::media::ItemData &itemData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint32_t serverTime = stream.Read<uint32_t>();
  std::cout << "serverTime: " << serverTime << '\n';
  uint32_t refObjID = stream.Read<uint32_t>();
  std::cout << "refObjID: " << refObjID << '\n';
  uint8_t scale = stream.Read<uint8_t>();
  std::cout << "scale: " << (int)scale << '\n';
  uint8_t curLevel = stream.Read<uint8_t>();
  std::cout << "curLevel: " << (int)curLevel << '\n';
  uint8_t maxLevel = stream.Read<uint8_t>();
  std::cout << "maxLevel: " << (int)maxLevel << '\n';
  uint64_t expOffset = stream.Read<uint64_t>();
  std::cout << "expOffset: " << expOffset << '\n';
  uint32_t sExpOffset = stream.Read<uint32_t>();
  std::cout << "sExpOffset: " << sExpOffset << '\n';
  uint64_t remainGold = stream.Read<uint64_t>();
  std::cout << "remainGold: " << remainGold << '\n';
  uint32_t remainSkillPoint = stream.Read<uint32_t>();
  std::cout << "remainSkillPoint: " << remainSkillPoint << '\n';
  uint16_t remainStatPoint = stream.Read<uint16_t>();
  std::cout << "remainStatPoint: " << remainStatPoint << '\n';
  uint8_t remainHwanCount = stream.Read<uint8_t>();
  std::cout << "remainHwanCount: " << (int)remainHwanCount << '\n';
  uint32_t gatheredExpPoint = stream.Read<uint32_t>();
  std::cout << "gatheredExpPoint: " << gatheredExpPoint << '\n';
  hp_ = stream.Read<uint32_t>();
  std::cout << "hp_: " << hp_ << '\n';
  mp_ = stream.Read<uint32_t>();
  std::cout << "mp_: " << mp_ << '\n';
  uint8_t autoInverstExp = stream.Read<uint8_t>();
  std::cout << "autoInverstExp: " << (int)autoInverstExp << '\n';
  uint8_t dailyPK = stream.Read<uint8_t>();
  std::cout << "dailyPK: " << (int)dailyPK << '\n';
  uint16_t totalPK = stream.Read<uint16_t>();
  std::cout << "totalPK: " << totalPK << '\n';
  uint32_t pKPenaltyPoint = stream.Read<uint32_t>();
  std::cout << "pKPenaltyPoint: " << pKPenaltyPoint << '\n';
  uint8_t hwanLevel = stream.Read<uint8_t>();
  std::cout << "hwanLevel: " << (int)hwanLevel << '\n';
  uint8_t freePVP = stream.Read<uint8_t>(); // 0 = None, 1 = Red, 2 = Gray, 3 = Blue, 4 = White, 5 = Gold
  std::cout << "freePVP: " << (int)freePVP << '\n';

  //=====================================================================================
  //===================================== Inventory =====================================
  //=====================================================================================

  uint8_t inventorySize = stream.Read<uint8_t>();
  std::cout << "inventorySize: " << (int)inventorySize << '\n';
  uint8_t inventoryItemCount = stream.Read<uint8_t>();
  std::cout << "inventoryItemCount: " << (int)inventoryItemCount << '\n';

  for (int itemNum=0; itemNum<inventoryItemCount; ++itemNum) {
    uint8_t slotNum = stream.Read<uint8_t>();
    uint32_t rentType = stream.Read<uint32_t>(); // TODO: Enum for this
    
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

    ItemVariantType parsedItem;

    uint32_t refItemId = stream.Read<uint32_t>();
    std::cout << "Item " << refItemId << " in slot " << (int)slotNum << ", with rentType: " << rentType << '\n';
    if (!itemData.haveItemWithId(refItemId)) {
      throw std::runtime_error("Unable to parse packet. Encountered an item (id:"+std::to_string(refItemId)+") for which we have no data on.");
    }
    const pk2::media::Item &item = itemData.getItemById(refItemId);
    std::cout << "typeId1: " << (int)item.typeId1 << ", ";
    std::cout << "typeId2: " << (int)item.typeId2 << ", ";
    std::cout << "typeId3: " << (int)item.typeId3 << ", ";
    std::cout << "typeId4: " << (int)item.typeId4 << '\n';

    if (item.typeId1 == 3) {
      if (item.typeId2 == 1) {
        // CGItemEquip
        parsedItem = parseItemEquipment(stream);
      } else if (item.typeId2 == 2) {
        if (item.typeId3 == 1) {                                
          // CGItemCOSSummoner
          if (item.typeId4 == 2) {
            parsedItem = parseItemCosGrowthSummoner(stream);
          } else {
            parsedItem = parseItemCosAbilitySummoner(stream);
          }
        } else if (item.typeId3 == 2) {
          // CGItemMonsterCapsule (rogue mask)
          parsedItem = parseItemMonsterCapsule(stream);
        } else if (item.typeId3 == 3) {
          // CGItemStorage
          parsedItem = parseItemStorage(stream);
        }
      } else if (item.typeId2 == 3) {
        // CGItemExpendable
        if (item.typeId3 == 11) {
          if (item.typeId4 == 1 || item.typeId4 == 2) {
            // MAGICSTONE, ATTRSTONE
            parsedItem = parseItemStone(stream);
          } else {
            // Other expendable
            parsedItem = parseItemExpendable(stream);
          }
        } else if (item.typeId3 == 14 && item.typeId4 == 2) {
          // Magic pop
          parsedItem = parseItemMagicPop(stream);
        } else {
          // Other expendable
          parsedItem = parseItemExpendable(stream);
        }
      }
    }

    // Set item's refItemId
    // TODO: Improve
    if (std::holds_alternative<item::ItemEquipment>(parsedItem)) {
      auto &specificItem = std::get<item::ItemEquipment>(parsedItem);
      specificItem.refItemId = refItemId;
      specificItem.itemInfo = &item;
      item::print(specificItem);
      std::cout << '\n';
    } else if (std::holds_alternative<item::ItemCosGrowthSummoner>(parsedItem)) {
      auto &specificItem = std::get<item::ItemCosGrowthSummoner>(parsedItem);
      specificItem.refItemId = refItemId;
      specificItem.itemInfo = &item;
      item::print(specificItem);
      std::cout << '\n';
    } else if (std::holds_alternative<item::ItemCosAbilitySummoner>(parsedItem)) {
      auto &specificItem = std::get<item::ItemCosAbilitySummoner>(parsedItem);
      specificItem.refItemId = refItemId;
      specificItem.itemInfo = &item;
      item::print(specificItem);
      std::cout << '\n';
    } else if (std::holds_alternative<item::ItemMonsterCapsule>(parsedItem)) {
      auto &specificItem = std::get<item::ItemMonsterCapsule>(parsedItem);
      specificItem.refItemId = refItemId;
      specificItem.itemInfo = &item;
      item::print(specificItem);
      std::cout << '\n';
    } else if (std::holds_alternative<item::ItemStorage>(parsedItem)) {
      auto &specificItem = std::get<item::ItemStorage>(parsedItem);
      specificItem.refItemId = refItemId;
      specificItem.itemInfo = &item;
      item::print(specificItem);
      std::cout << '\n';
    } else if (std::holds_alternative<item::ItemExpendable>(parsedItem)) {
      auto &specificItem = std::get<item::ItemExpendable>(parsedItem);
      specificItem.refItemId = refItemId;
      specificItem.itemInfo = &item;
      item::print(specificItem);
      std::cout << '\n';
    } else if (std::holds_alternative<item::ItemStone>(parsedItem)) {
      auto &specificItem = std::get<item::ItemStone>(parsedItem);
      specificItem.refItemId = refItemId;
      specificItem.itemInfo = &item;
      item::print(specificItem);
      std::cout << '\n';
    } else if (std::holds_alternative<item::ItemMagicPop>(parsedItem)) {
      auto &specificItem = std::get<item::ItemMagicPop>(parsedItem);
      specificItem.refItemId = refItemId;
      specificItem.itemInfo = &item;
      item::print(specificItem);
      std::cout << '\n';
    }

    inventoryItemMap_.insert({slotNum, parsedItem});

    /*if (item.typeId1 == 3) {
      std::cout << "item.typeId1 == 3\n";
      // CGItem
      // ITEM_
      if (item.typeId2 == 1) {
        std::cout << "item.typeId2 == 1\n";
        // CGItemEquip
        // ITEM_CH
        // ITEM_EU
        // AVATAR_
        uint8_t optLevel = stream.Read<uint8_t>();
        uint64_t variance = stream.Read<uint64_t>();
        uint32_t durability = stream.Read<uint32_t>(); // "Data"
        uint8_t magParamNum = stream.Read<uint8_t>();

        for (int paramIndex=0; paramIndex<magParamNum; ++paramIndex) {
          uint32_t type = stream.Read<uint32_t>();
          uint32_t value = stream.Read<uint32_t>();
        }
        
        uint8_t bindingOptionType1 = stream.Read<uint8_t>(); // 1 = Socket
        uint8_t bindingOptionCount1 = stream.Read<uint8_t>();
        for (int bindingOptionIndex=0; bindingOptionIndex<bindingOptionCount1; ++bindingOptionIndex) {
          uint8_t slot = stream.Read<uint8_t>();
          uint32_t id = stream.Read<uint32_t>();
          uint32_t nParam1 = stream.Read<uint32_t>();
        }
        
        uint8_t bindingOptionType2 = stream.Read<uint8_t>(); // 2 = Advanced elixir
        uint8_t bindingOptionCount2 = stream.Read<uint8_t>();
        for (int bindingOptionIndex=0; bindingOptionIndex<bindingOptionCount2; ++bindingOptionIndex) {
          uint8_t slot = stream.Read<uint8_t>();
          uint32_t id = stream.Read<uint32_t>();
          uint32_t optValue = stream.Read<uint32_t>();
        }
      } else if (item.typeId2 == 2) {
        // CGItemContainer
        if (item.typeId3 == 1) {                                
          // CGItemCOSSummoner
          // ITEM_COS_P
          uint8_t lifeState = stream.Read<uint8_t>(); // 1=inactive, 2=summoned, 3=active, 4=dead
          if (lifeState == 1) { // "Embryo"/inactive
            // Nothing to read
          } else {
            uint32_t refObjID = stream.Read<uint32_t>();
            uint16_t nameLength = stream.Read<uint16_t>();
            std::string name = stream.Read_Ascii(nameLength);
            if (item.typeId4 == 2) {
              //ITEM_COS_P (Ability)
              uint32_t secondsToRentEndTime = stream.Read<uint32_t>();
            }
            uint8_t timedJobCount = stream.Read<uint8_t>();
            for (int jobNum=0; jobNum<timedJobCount; ++jobNum) {
              uint8_t category = stream.Read<uint8_t>();
              uint32_t jobId = stream.Read<uint32_t>(); // category3 = RefSkillId, category5 = RefItemId
              uint32_t timeToKeep = stream.Read<uint32_t>();
              if (category == 5) {
                uint32_t data1 = stream.Read<uint32_t>();
                uint8_t data2 = stream.Read<uint8_t>();
              }
            }
          }
        } else if (item.typeId3 == 2) {
          // CGItemMonsterCapsule (rogue mask)
          // ITEM_ETC_TRANS_MONSTER
          uint32_t refObjID = stream.Read<uint32_t>();
        } else if (item.typeId3 == 3) {
          // CGItemStorage
          // MAGIC_CUBE
          uint32_t quantity = stream.Read<uint32_t>(); // Do not confuse with StackCount, this indicates the amount of elixirs in the cube
        }
      } else if (item.typeId2 == 3) {
        // CGItemExpendable
        // ITEM_ETC
        uint16_t stackCount = stream.Read<uint16_t>();
        if (item.typeId3 == 11) {
          if (item.typeId4 == 1 || item.typeId4 == 2) {
            // MAGICSTONE, ATTRSTONE
            uint8_t attributeAssimilationProbability = stream.Read<uint8_t>(); // stored in _Items.OptLevel on the server side
          }
        } else if (item.typeId3 == 14 && item.typeId4 == 2) {
          // Magic pop
          // ITEM_MALL_GACHA_CARD_WIN
          // ITEM_MALL_GACHA_CARD_LOSE
          uint8_t magParamCount = stream.Read<uint8_t>();
          for (int paramIndex=0; paramIndex<magParamCount; ++paramIndex) {
              uint32_t type = stream.Read<uint32_t>();
              uint32_t value = stream.Read<uint32_t>();
          }
        }
      }
    }*/
  }

  //=====================================================================================
  //================================== Avatar Inventory =================================
  //=====================================================================================

  uint8_t avatarInventorySize = stream.Read<uint8_t>();
  uint8_t avatarItemCount = stream.Read<uint8_t>();
  std::cout << "avatarInventorySize: " << (int)avatarInventorySize << ", ";
  std::cout << "avatarItemCount: " << (int)avatarItemCount << '\n';

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
    std::cout << "Avatar " << refItemId << " in slot " << (int)slotNum << ", with rentType: " << rentType << '\n';
    if (!itemData.haveItemWithId(refItemId)) {
      throw std::runtime_error("Unable to parse packet. Encountered an item (id:"+std::to_string(refItemId)+") for which we have no data on.");
    }
    const pk2::media::Item &item = itemData.getItemById(refItemId);
    std::cout << "typeId1: " << (int)item.typeId1 << ", ";
    std::cout << "typeId2: " << (int)item.typeId2 << ", ";
    std::cout << "typeId3: " << (int)item.typeId3 << ", ";
    std::cout << "typeId4: " << (int)item.typeId4 << '\n';
      
    if (item.typeId1 == 3) {
      // ITEM_
      if (item.typeId2 == 1) { //TODO: Narrow filters for AvatarInventory
        // ITEM_CH
        // ITEM_EU
        // AVATAR_
        item::ItemEquipment avatarItem = parseItemEquipment(stream);
        // uint8_t optLevel = stream.Read<uint8_t>();
        // uint64_t variance = stream.Read<uint64_t>();
        // uint32_t durability = stream.Read<uint32_t>(); //"Data"
        // uint8_t magParamNum = stream.Read<uint8_t>();

        // for (int paramIndex=0; paramIndex<magParamNum; ++paramIndex) {
        //   uint32_t type = stream.Read<uint32_t>();
        //   uint32_t value = stream.Read<uint32_t>();
        // }
        
        // uint8_t bindingOptionType1 = stream.Read<uint8_t>(); // 1 = Socket
        // uint8_t bindingOptionCount1 = stream.Read<uint8_t>();
        // for (int bindingOptionIndex=0; bindingOptionIndex<bindingOptionCount1; ++bindingOptionIndex) {
        //   uint8_t slot = stream.Read<uint8_t>();
        //   uint32_t id = stream.Read<uint32_t>();
        //   uint32_t nParam1 = stream.Read<uint32_t>();
        // }
        
        // uint8_t bindingOptionType2 = stream.Read<uint8_t>(); // 2 = Advanced elixir
        // uint8_t bindingOptionCount2 = stream.Read<uint8_t>();
        // for (int bindingOptionIndex=0; bindingOptionIndex<bindingOptionCount2; ++bindingOptionIndex) {
        //   uint8_t slot = stream.Read<uint8_t>();
        //   uint32_t id = stream.Read<uint32_t>();
        //   uint32_t optValue = stream.Read<uint32_t>();
        // }        
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
    std::cout << "Mastery (id:" << id << ") is level " << (int)level << '\n';
    hasNextMastery = stream.Read<uint8_t>();
  }

  uint8_t unknownByte1 = stream.Read<uint8_t>(); // "not a counter"

  //=====================================================================================
  //====================================== Skills =======================================
  //=====================================================================================

  uint8_t hasNextSkill = stream.Read<uint8_t>();
  std::cout << "Skills: [";
  while (hasNextSkill == 1) {
    uint32_t id = stream.Read<uint32_t>();
    uint8_t enabled = stream.Read<uint8_t>();

    if (enabled != 0) {
      std::cout << id << ',';
    }
    hasNextSkill = stream.Read<uint8_t>();
  }
  std::cout << "]\n";

  //=====================================================================================
  //====================================== Quests =======================================
  //=====================================================================================

  uint16_t completedQuestCount = stream.Read<uint16_t>();
  std::cout << completedQuestCount << " completed quest(s): [";

  for (int i=0; i<completedQuestCount; ++i) {
    uint32_t quest = stream.Read<uint32_t>();
    std::cout << quest << ',';
  }
  std::cout << "]\n";

  uint8_t activeQuestCount = stream.Read<uint8_t>();
  std::cout << (int)activeQuestCount << " active quest(s): [\n";

  for (int i=0; i<activeQuestCount; ++i) {
    uint32_t refQuestID = stream.Read<uint32_t>();
    uint8_t achivementCount = stream.Read<uint8_t>();
    uint8_t requiresAutoShareParty = stream.Read<uint8_t>();
    uint8_t type = stream.Read<uint8_t>();
    std::cout << "  (" << refQuestID << ',' << (int)type << ')';

    if (type == 28) {
      uint32_t remainingTime = stream.Read<uint32_t>();
      std::cout << ", remaining time:" << remainingTime;
    }
    uint8_t questStatus = stream.Read<uint8_t>();
    std::cout << ", status:" << (int)questStatus;
    
    if (type != 8) {
      uint8_t objectiveCount = stream.Read<uint8_t>();
      std::cout << ", objective count:" << (int)objectiveCount;

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
    std::cout << '\n';
  }
  std::cout << "]\n";

  uint8_t unknownByte2 = stream.Read<uint8_t>(); // "Structure changes!!!"

  //=====================================================================================
  //================================== Collection Book ==================================
  //=====================================================================================

  uint32_t collectionBookStartedThemeCount = stream.Read<uint32_t>();
  std::cout << "collectionBookStartedThemeCount: " << collectionBookStartedThemeCount << " [";

  for (uint32_t i=0; i<collectionBookStartedThemeCount; ++i) {
    uint32_t themeIndex = stream.Read<uint32_t>();
    uint32_t themeStartedDateTime = stream.Read<uint32_t>(); // SROTimeStamp
    uint32_t themePages = stream.Read<uint32_t>();
    std::cout << "  themeIndex: " << themeIndex;
    std::cout << ", themeStartedDateTime: " << themeStartedDateTime;
    std::cout << ", themePages: " << themePages << '\n';
  }
  std::cout << "]\n";
  
  entityUniqueId_ = stream.Read<uint32_t>();
  std::cout << "entityUniqueId_: " << entityUniqueId_ << '\n';

  //=====================================================================================
  //===================================== Position ======================================
  //=====================================================================================

  // //Position
  uint16_t regionId = stream.Read<uint16_t>();
  float posX = stream.Read<float>();
  float posY = stream.Read<float>();
  float posZ = stream.Read<float>();
  uint16_t angle = stream.Read<uint16_t>();
  std::cout << "regionId: " << regionId << '\n';
  std::cout << "posX: " << posX << '\n';
  std::cout << "posY: " << posY << '\n';
  std::cout << "posZ: " << posZ << '\n';
  std::cout << "angle: " << angle << '\n';

  //=====================================================================================
  //===================================== Movement ======================================
  //=====================================================================================

  uint8_t hasDestination = stream.Read<uint8_t>();
  uint8_t movementType = stream.Read<uint8_t>();
  std::cout << "hasDestination: " << (int)hasDestination << '\n';
  std::cout << "movementType: " << (int)movementType << '\n';

  if (hasDestination) {
    uint16_t destinationRegion = stream.Read<uint16_t>();
    std::cout << "destinationRegion: " << (int)destinationRegion << '\n';

    if (regionId < std::numeric_limits<uint16_t>::max()) {
      // World
      uint16_t destinationX = stream.Read<uint16_t>();
      std::cout << "destinationX: " << (int)destinationX << '\n';
      uint16_t destinationY = stream.Read<uint16_t>();
      std::cout << "destinationY: " << (int)destinationY << '\n';
      uint16_t destinationZ = stream.Read<uint16_t>();
      std::cout << "destinationZ: " << (int)destinationZ << '\n';
    } else {
      // Dungeon
      uint32_t destinationOffsetX = stream.Read<uint32_t>();
      std::cout << "destinationOffsetX: " << (int)destinationOffsetX << '\n';
      uint32_t destinationOffsetY = stream.Read<uint32_t>();
      std::cout << "destinationOffsetY: " << (int)destinationOffsetY << '\n';
      uint32_t destinationOffsetZ = stream.Read<uint32_t>();
      std::cout << "destinationOffsetZ: " << (int)destinationOffsetZ << '\n';
    }
  } else {
    uint8_t source = stream.Read<uint8_t>(); // 0 = Spinning, 1 = Sky-/Key-walking
    std::cout << "source: " << (int)source << '\n';
    uint16_t angle = stream.Read<uint16_t>(); // Represents the new angle, character is looking at
    std::cout << "angle: " << (int)angle << '\n';
  }

  //=====================================================================================
  //======================================= State =======================================
  //=====================================================================================

  uint8_t lifeState = stream.Read<uint8_t>(); // 1 = Alive, 2 = Dead
  std::cout << "lifeState: " << (int)lifeState << '\n';
  uint8_t unkByte0 = stream.Read<uint8_t>();
  std::cout << "unkByte0: " << (int)unkByte0 << '\n';
  uint8_t motionState = stream.Read<uint8_t>(); // 0 = None, 2 = Walking, 3 = Running, 4 = Sitting
  std::cout << "motionState: " << (int)motionState << '\n';
  uint8_t status = stream.Read<uint8_t>(); // 0 = None, 1 = Hwan, 2 = Untouchable, 3 = GameMasterInvincible, 5 = GameMasterInvisible, 5 = ?, 6 = Stealth, 7 = Invisible
  std::cout << "status: " << (int)status << '\n';
  float walkSpeed = stream.Read<float>();
  std::cout << "walkSpeed: " << walkSpeed << '\n';
  float runSpeed = stream.Read<float>();
  std::cout << "runSpeed: " << runSpeed << '\n';
  float hwanSpeed = stream.Read<float>();
  std::cout << "hwanSpeed: " << hwanSpeed << '\n';
  uint8_t buffCount = stream.Read<uint8_t>();
  std::cout << "buffCount: " << (int)buffCount << '\n';
  for (int i=0; i<buffCount; ++i) {
    uint32_t refSkillId = stream.Read<uint32_t>();
    uint32_t duration = stream.Read<uint32_t>();
    std::cout << "Buff #" << i << ", refSkillId: " << refSkillId << ", duration: " << duration << '\n';
    // TODO
    // if(skill.Params.Contains(1701213281)) {
    //   //1701213281 -> atfe -> "auto transfer effect" like Recovery Division
    //   uint8_t isCreator = stream.Read<uint8_t>();
    // }
  }

  uint16_t nameLength = stream.Read<uint16_t>();
  std::string name = stream.Read_Ascii(nameLength);
  std::cout << "name: \"" << name << "\"\n";
  uint16_t jobNameLength = stream.Read<uint16_t>();
  std::string jobName = stream.Read_Ascii(jobNameLength);
  std::cout << "jobName: \"" << jobName << "\"\n";
  uint8_t jobType = stream.Read<uint8_t>();
  std::cout << "jobType: " << (int)jobType << '\n';
  uint8_t jobLevel = stream.Read<uint8_t>();
  std::cout << "jobLevel: " << (int)jobLevel << '\n';
  uint32_t jobExp = stream.Read<uint32_t>();
  std::cout << "jobExp: " << jobExp << '\n';
  uint32_t jobContribution = stream.Read<uint32_t>();
  std::cout << "jobContribution: " << jobContribution << '\n';
  uint32_t jobReward = stream.Read<uint32_t>();
  std::cout << "jobReward: " << jobReward << '\n';
  uint8_t pvpState = stream.Read<uint8_t>(); // 0 = White, 1 = Purple, 2 = Red
  std::cout << "pvpState: " << (int)pvpState << '\n';
  uint8_t transportFlag = stream.Read<uint8_t>();
  std::cout << "transportFlag: " << (int)transportFlag << '\n';
  uint8_t inCombat = stream.Read<uint8_t>();
  std::cout << "inCombat: " << (int)inCombat << '\n';

  if (transportFlag == 1) {
    uint32_t transportId = stream.Read<uint32_t>();
  }

  uint8_t pvpFlag = stream.Read<uint8_t>(); // 0 = Red Side, 1 = Blue Side, 0xFF = None
  std::cout << "pvpFlag: " << (int)pvpFlag << '\n';
  uint64_t guideFlag = stream.Read<uint64_t>();
  std::cout << "guideFlag: " << guideFlag << '\n';
  uint32_t jId = stream.Read<uint32_t>();
  std::cout << "jId: " << jId << '\n';
  uint8_t gmFlag = stream.Read<uint8_t>();
  std::cout << "gmFlag: " << (int)gmFlag << '\n';

  uint8_t activationFlag = stream.Read<uint8_t>(); // ConfigType:0 --> (0 = Not activated, 7 = activated)
  std::cout << "activationFlag: " << (int)activationFlag << '\n';
  uint8_t hotkeyCount = stream.Read<uint8_t>(); // ConfigType:1
  std::cout << "hotkeyCount: " << (int)hotkeyCount << '\n';
  
  for (int i=0; i<hotkeyCount; ++i) {
    uint8_t slotSeq = stream.Read<uint8_t>();
    uint8_t slotContentType = stream.Read<uint8_t>();
    uint32_t slotData = stream.Read<uint32_t>();
    std::cout << "Hotkey #" << i << ", slotSeq: " << (int)slotSeq << ", slotContentType: " << (int)slotContentType << ", slotData: " << slotData << '\n';
  }

  uint16_t autoHPConfig = stream.Read<uint16_t>(); // ConfigType:11
  std::cout << "autoHPConfig: " << autoHPConfig << '\n';
  uint16_t autoMPConfig = stream.Read<uint16_t>(); // ConfigType:12
  std::cout << "autoMPConfig: " << autoMPConfig << '\n';
  uint16_t autoUniversalConfig = stream.Read<uint16_t>(); // ConfigType:13
  std::cout << "autoUniversalConfig: " << autoUniversalConfig << '\n';
  uint8_t autoPotionDelay = stream.Read<uint8_t>(); // ConfigType:14
  std::cout << "autoPotionDelay: " << (int)autoPotionDelay << '\n';

  uint8_t blockedWhisperCount = stream.Read<uint8_t>();
  std::cout << "blockedWhisperCount: " << (int)blockedWhisperCount << '\n';

  for (int i=0; i<blockedWhisperCount; ++i) {
    uint16_t targetLength = stream.Read<uint16_t>();
    std::string target = stream.Read_Ascii(targetLength);
    std::cout << "Blocked whisper #" << i << "\"" << target << "\"\n";
  }

  uint32_t unknownShort0 = stream.Read<uint32_t>(); //Structure changes!!!
  uint8_t unknownByte3 = stream.Read<uint8_t>(); //Structure changes!!!
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
} // namespace packet::parsing