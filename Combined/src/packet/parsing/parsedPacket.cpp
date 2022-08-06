#include "commonParsing.hpp"
#include "logging.hpp"
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

ParsedServerAgentEntityUpdateStatus::ParsedServerAgentEntityUpdateStatus(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;

  entityUniqueId_ = stream.Read<uint32_t>();
  updateFlag_ = static_cast<packet::enums::UpdateFlag>(stream.Read<uint16_t>());
  vitalBitmask_ = stream.Read<uint8_t>();

  if (vitalBitmask_ & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoHp)) {
    newHpValue_ = stream.Read<uint32_t>();
  }

  if (vitalBitmask_ & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoMp)) {
    newMpValue_ = stream.Read<uint32_t>();
  }

  if (vitalBitmask_ & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoHgp)) {
    newHgpValue_ = stream.Read<uint16_t>();
  }

  if (vitalBitmask_ & static_cast<uint8_t>(packet::enums::VitalInfoFlag::kVitalInfoAbnormal)) {
    stateBitmask_ = stream.Read<uint32_t>();
    for (uint32_t i=0; i<32; ++i) {
      const auto bit = (1 << i);
      if (bit > static_cast<uint32_t>(packet::enums::AbnormalStateFlag::kZombie) && stateBitmask_ & bit) {
        stateLevels_.push_back(stream.Read<uint8_t>());
      }
    }
  }
}

uint32_t ParsedServerAgentEntityUpdateStatus::entityUniqueId() const {
  return entityUniqueId_;
}

packet::enums::UpdateFlag ParsedServerAgentEntityUpdateStatus::updateFlag() const {
  return updateFlag_;
}

uint8_t ParsedServerAgentEntityUpdateStatus::vitalBitmask() const {
  return vitalBitmask_;
}

uint32_t ParsedServerAgentEntityUpdateStatus::newHpValue() const {
  return newHpValue_;
}

uint32_t ParsedServerAgentEntityUpdateStatus::newMpValue() const {
  return newMpValue_;
}

uint16_t ParsedServerAgentEntityUpdateStatus::newHgpValue() const {
  return newHgpValue_;
}

uint32_t ParsedServerAgentEntityUpdateStatus::stateBitmask() const {
  return stateBitmask_;
}

const std::vector<uint8_t>& ParsedServerAgentEntityUpdateStatus::stateLevels() const {
  return stateLevels_;
}

//=========================================================================================================================================================

uint32_t ParsedServerAgentAbnormalInfo::stateBitmask() const {
  return stateBitmask_;
}

const std::array<packet::structures::vitals::AbnormalState, 32>& ParsedServerAgentAbnormalInfo::states() const {
  return states_;
}

ParsedServerAgentAbnormalInfo::ParsedServerAgentAbnormalInfo(const PacketContainer &packet) : ParsedPacket(packet) { 
  StreamUtility stream = packet.data;
  stateBitmask_ = stream.Read<uint32_t>();
  for (uint32_t i=0; i<32; ++i) {
    const auto bit = (1 << i);
    if (stateBitmask_ & bit) {
      auto &state = states_[i];
      state.totalTime = stream.Read<uint32_t>();
      state.timeElapsed = stream.Read<uint16_t>();
      if (bit <= static_cast<uint32_t>(packet::enums::AbnormalStateFlag::kZombie)) {
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

uint8_t ParsedServerAgentInventoryItemUseResponse::result() const {
  return result_;
}

uint8_t ParsedServerAgentInventoryItemUseResponse::slotNum() const {
  return slotNum_;
}

uint16_t ParsedServerAgentInventoryItemUseResponse::remainingCount() const {
  return remainingCount_;
}

uint16_t ParsedServerAgentInventoryItemUseResponse::itemData() const {
  return itemData_;
}

packet::enums::InventoryErrorCode ParsedServerAgentInventoryItemUseResponse::errorCode() const {
  return errorCode_;
}

ParsedServerAgentInventoryItemUseResponse::ParsedServerAgentInventoryItemUseResponse(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;

  result_ = stream.Read<uint8_t>();
  if (result_ == 1) {
    // Success
    slotNum_ = stream.Read<uint8_t>();
    remainingCount_ = stream.Read<uint16_t>();
    itemData_ = stream.Read<uint16_t>();
  } else {
    errorCode_ = static_cast<packet::enums::InventoryErrorCode>(stream.Read<uint16_t>());
  }
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

std::shared_ptr<Object> newObjectFromId(uint32_t refObjId, const pk2::CharacterData &characterData, const pk2::ItemData &itemData, const pk2::TeleportData &teleportData) {
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
                                   const pk2::CharacterData &characterData,
                                   const pk2::ItemData &itemData,
                                   const pk2::SkillData &skillData,
                                   const pk2::TeleportData &teleportData) {
  const uint32_t refObjId = stream.Read<uint32_t>();
  if (refObjId == std::numeric_limits<uint32_t>::max()) {
    // Special case, refObjId == -1
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
    return {};
  }

  std::shared_ptr<Object> obj = newObjectFromId(refObjId, characterData, itemData, teleportData);
  if (!obj) {
    throw std::runtime_error("Failed to create object for spawn");
  }

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
    } else if (character.typeId2 == 2 && character.typeId3 == 5) {
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
      packet::enums::AngleAction angleAction_ = static_cast<packet::enums::AngleAction>(stream.Read<uint8_t>());
      uint16_t angle = stream.Read<uint16_t>(); // Represents the new angle, character is looking at
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
      // } else if (character.typeId3 == 5) {
      //   // CGObjSiegeStruct
      //   uint32_t unk0 = stream.Read<uint32_t>(); // 0xFFFFFFFF
      //   uint16_t unk1 = stream.Read<uint16_t>(); // 0x0054
      //   uint32_t unk2 = stream.Read<uint32_t>(); // 0x000052FE
      //   uint32_t unk3 = stream.Read<uint32_t>(); // 0x0001ACB9
      //   uint16_t unk4 = stream.Read<uint16_t>(); // 0x4547      (region Id)
      //   uint32_t xBytes = stream.Read<uint32_t>();
      //   float x = *reinterpret_cast<float*>(&xBytes);
      //   uint32_t yBytes = stream.Read<uint32_t>();
      //   float y = *reinterpret_cast<float*>(&yBytes);
      //   uint32_t zBytes = stream.Read<uint32_t>();
      //   float z = *reinterpret_cast<float*>(&zBytes);
      //   uint16_t angle = stream.Read<uint16_t>();
      }
    }
  } else if (itemData.haveItemWithId(obj->refObjId) && itemData.getItemById(obj->refObjId).typeId1 == 3) {
    Item *itemPtr = dynamic_cast<Item*>(obj.get());
    if (itemPtr == nullptr) {
      throw std::runtime_error("parseSpawn, have an item, but the obj pointer cannot be cast to a Item");
    }
    const auto &item = itemData.getItemById(obj->refObjId);
    std::cout << "Item with refid " << obj->refObjId << " spawned\n";
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
      } else if (item.typeId3 == 8 || item.typeId3 == 9) {
        // ITEM_ETC_TRADE
        // ITEM_ETC_QUEST
        uint16_t ownerNameLength = stream.Read<uint16_t>();
        std::string ownerName = stream.Read_Ascii(ownerNameLength);
        std::cout << "Item is a quest item belonging to " << ownerName << '\n';
      }
    }
    itemPtr->gId = stream.Read<uint32_t>();
    std::cout << "Item's GID is " << itemPtr->gId << '\n';
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
  }
  return obj;
}

uint32_t parseDespawn(StreamUtility &stream) {
  return stream.Read<uint32_t>();
}

ParsedServerAgentEntityGroupSpawnData::ParsedServerAgentEntityGroupSpawnData(const PacketContainer &packet,
                                                         const pk2::CharacterData &characterData,
                                                         const pk2::ItemData &itemData,
                                                         const pk2::SkillData &skillData,
                                                         const pk2::TeleportData &teleportData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  // This data is originally from the begin packet (before the data packet)
  groupSpawnType_ = static_cast<GroupSpawnType>(stream.Read<uint8_t>());
  uint16_t groupSpawnAmount = stream.Read<uint16_t>();
  if (groupSpawnType_ == GroupSpawnType::kSpawn) {
    for (int spawnNum=0; spawnNum<groupSpawnAmount; ++spawnNum) {
      auto obj = parseSpawn(stream, characterData, itemData, skillData, teleportData);
      if (obj) {
        // TODO: Handle "skill objects", like the recovery circle (will be nullptr)
        objects_.emplace_back(obj);
      }
    }
  } else if (groupSpawnType_ == GroupSpawnType::kDespawn) {
    for (int despawnNum=0; despawnNum<groupSpawnAmount; ++despawnNum) {
      despawns_.emplace_back(parseDespawn(stream));
    }
  }
}

GroupSpawnType ParsedServerAgentEntityGroupSpawnData::groupSpawnType() const {
  return groupSpawnType_;
}

const std::vector<std::shared_ptr<Object>>& ParsedServerAgentEntityGroupSpawnData::objects() const {
  return objects_;
}

const std::vector<uint32_t>& ParsedServerAgentEntityGroupSpawnData::despawns() const {
  return despawns_;
}

//=========================================================================================================================================================

ParsedServerAgentSpawn::ParsedServerAgentSpawn(const PacketContainer &packet,
                                               const pk2::CharacterData &characterData,
                                               const pk2::ItemData &itemData,
                                               const pk2::SkillData &skillData,
                                               const pk2::TeleportData &teleportData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  object_ = parseSpawn(stream, characterData, itemData, skillData, teleportData);
  if (object_) {
    // TODO: Handle "skill objects", like the recovery circle (will be nullptr)
    if (object_->typeId1 == 1 || object_->typeId1 == 4) {
      //BIONIC and STORE
      uint8_t spawnType = stream.Read<uint8_t>(); // 1=COS_SUMMON, 3=SPAWN, 4=SPAWN_WALK
    } else if (object_->typeId1 == 3) {
      uint8_t dropSource = stream.Read<uint8_t>();
      uint32_t dropperUniqueId = stream.Read<uint32_t>();
    }
  }
}

std::shared_ptr<Object> ParsedServerAgentSpawn::object() const {
  return object_;
}
                         
//=========================================================================================================================================================

ParsedServerAgentDespawn::ParsedServerAgentDespawn(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  LOG() << "Parsing a despawn" << std::endl;
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
  // 0x01 = success, 0x02 = error
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
  action_ = static_cast<packet::enums::CharacterSelectionAction>(stream.Read<uint8_t>());
  result_ = stream.Read<uint8_t>();
  if (result_ == 0x01 && action_ == packet::enums::CharacterSelectionAction::kList) {
    // Listing characters
    const uint8_t kCharCount = stream.Read<uint8_t>();
    for (int i=0; i<kCharCount; ++i) {
      packet::structures::CharacterSelection::Character character;
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
        packet::structures::CharacterSelection::Item item;   
        item.refId = stream.Read<uint32_t>();
        item.plus = stream.Read<uint8_t>();
        character.items.emplace_back(std::move(item));
      }
      const uint8_t kAvatarCount = stream.Read<uint8_t>();
      for (int j=0; j<kAvatarCount; ++j) {
        packet::structures::CharacterSelection::Avatar avatar;   
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

packet::enums::CharacterSelectionAction ParsedServerAgentCharacterSelectionActionResponse::action() const {
  return action_;
}

uint8_t ParsedServerAgentCharacterSelectionActionResponse::result() const {
  return result_;
}

const std::vector<packet::structures::CharacterSelection::Character>& ParsedServerAgentCharacterSelectionActionResponse::characters() const {
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
  // 0x01 = success, 0x02 = error
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
  result_ = static_cast<packet::enums::LoginResult>(stream.Read<uint8_t>());
  if (result_ == packet::enums::LoginResult::kSuccess) {
    token_ = stream.Read<uint32_t>();
    uint16_t ipLength = stream.Read<uint16_t>();
    std::string ip = stream.Read_Ascii(ipLength);
    uint16_t port = stream.Read<uint16_t>();
  } else if (result_ == packet::enums::LoginResult::kFailed) {
    uint8_t errorCode = stream.Read<uint8_t>();
    if (errorCode == 0x01) {
      uint32_t maxAttempts = stream.Read<uint32_t>();
      uint32_t currentAttempts = stream.Read<uint32_t>();
    } else if (errorCode == 0x02) {
      packet::enums::LoginBlockType blockType = static_cast<packet::enums::LoginBlockType>(stream.Read<uint8_t>());
      if (blockType == packet::enums::LoginBlockType::kPunishment) {
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
  } else if (result_ == packet::enums::LoginResult::kOther) {
    /* uint8_t unkByte0 = */ stream.Read<uint8_t>();
    /* uint8_t unkByte1 = */ stream.Read<uint8_t>();
    uint16_t messageLength = stream.Read<uint16_t>();
    /* std::string message = */ stream.Read_Ascii(messageLength);
    /* uint16_t unkUShort0 = */ stream.Read<uint16_t>();
  }
}

packet::enums::LoginResult ParsedLoginResponse::result() const {
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

ParsedClientItemMove::ParsedClientItemMove(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  movement_.type = static_cast<packet::enums::ItemMovementType>(stream.Read<uint8_t>());
  if (movement_.type == packet::enums::ItemMovementType::kWithinInventory) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kWithinStorage ||
             movement_.type == packet::enums::ItemMovementType::kWithinGuildStorage) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
    uint32_t unk0 = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kInventoryToStorage ||
             movement_.type == packet::enums::ItemMovementType::kStorageToInventory ||
             movement_.type == packet::enums::ItemMovementType::kInventoryToGuildStorage ||
             movement_.type == packet::enums::ItemMovementType::kGuildStorageToInventory) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint32_t unk0 = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kBuyFromNPC) {
    movement_.storeTabNumber = stream.Read<uint8_t>();
    movement_.storeSlotNumber = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
    movement_.globalId = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kSellToNPC) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
    uint32_t unk1 = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kGoldDrop ||
             movement_.type == packet::enums::ItemMovementType::kGoldStorageWithdraw ||
             movement_.type == packet::enums::ItemMovementType::kGoldStorageDeposit ||
             movement_.type == packet::enums::ItemMovementType::kGoldGuildStorageDeposit ||
             movement_.type == packet::enums::ItemMovementType::kGoldGuildStorageWithdraw) {
    uint64_t goldAmount = stream.Read<uint64_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kCosToInventory ||
             movement_.type == packet::enums::ItemMovementType::kInventoryToCos) {
    uint32_t unk4 = stream.Read<uint32_t>();
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kWithinCos) {
    uint32_t unk4 = stream.Read<uint32_t>();
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kAvatarToInventory) {
    uint8_t sourceAvatarInventorySlot = stream.Read<uint8_t>();
    uint8_t destInventorySlot = stream.Read<uint8_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kInventoryToAvatar) {
    uint8_t sourceInventorySlot = stream.Read<uint8_t>();
    uint8_t destAvatarInventorySlot = stream.Read<uint8_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kDropItem) {
    uint8_t sourceInventorySlot = stream.Read<uint8_t>();
  } else {
    std::cout << "New item movement type! " << static_cast<int>(movement_.type) << '\n';
    std::cout << "Dump: " << DumpToString(stream) << '\n';
  }
}

structures::ItemMovement ParsedClientItemMove::movement() const {
  return movement_;
}

void printObj(const packet::parsing::Object *obj, const pk2::GameData &gameData) {
  switch (obj->type) {
    case packet::parsing::ObjectType::kPlayerCharacter:
      {
        auto ptr = reinterpret_cast<const packet::parsing::PlayerCharacter*>(obj);
        printf("%7s %5d %5d (%8.2f,%8.2f,%8.2f) GId:%d, name:\"%s\"\n","Player", obj->gId, obj->refObjId, ptr->x, ptr->y, ptr->z, ptr->gId, ptr->name.c_str());
      } 
      break;
    case packet::parsing::ObjectType::kNonplayerCharacter:
      {
        auto ptr = reinterpret_cast<const packet::parsing::NonplayerCharacter*>(obj);
        const auto &character = gameData.characterData().getCharacterById(obj->refObjId);
        printf("%7s %5d %5d (%8.2f,%8.2f,%8.2f) \"%s\"\n","NPC", obj->gId, obj->refObjId, ptr->x, ptr->y, ptr->z, character.codeName128.c_str());
      } 
      break;
    case packet::parsing::ObjectType::kMonster:
      {
        auto ptr = reinterpret_cast<const packet::parsing::Monster*>(obj);
        const auto &character = gameData.characterData().getCharacterById(obj->refObjId);
        printf("%7s %5d %5d (%8.2f,%8.2f,%8.2f) type:%d, \"%s\"\n","Monster", obj->gId, obj->refObjId, ptr->x, ptr->y, ptr->z, ptr->monsterRarity, character.codeName128.c_str());
      } 
      break;
    case packet::parsing::ObjectType::kItem:
      {
        auto ptr = reinterpret_cast<const packet::parsing::Item*>(obj);
        const auto &item = gameData.itemData().getItemById(obj->refObjId);
        printf("%7s %5d %5d (%8.2f,%8.2f,%8.2f) rarity:%d, \"%s\"\n","Item", obj->gId, obj->refObjId, ptr->x, ptr->y, ptr->z, ptr->rarity, item.codeName128.c_str());
      } 
      break;
    case packet::parsing::ObjectType::kPortal:
      {
        auto ptr = reinterpret_cast<const packet::parsing::Portal*>(obj);
        const auto &portal = gameData.teleportData().getTeleportById(obj->refObjId);
        printf("%7s %5d %5d (%8.2f,%8.2f,%8.2f) \"%s\"\n","Portal", obj->gId, obj->refObjId, ptr->x, ptr->y, ptr->z, portal.codeName128.c_str());
      }
      break;
  }
}

//=========================================================================================================================================================
} // namespace packet::parsing