#include "commonParsing.hpp"
#include "serverAgentCharacterData.hpp"

#include <iostream>

namespace packet::parsing {

ParsedServerAgentCharacterData::ParsedServerAgentCharacterData(const PacketContainer &packet, const pk2::ItemData &itemData, const pk2::SkillData &skillData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint32_t serverTime = stream.Read<uint32_t>();
  refObjId_ = stream.Read<uint32_t>();
  uint8_t scale = stream.Read<uint8_t>();
  uint8_t curLevel = stream.Read<uint8_t>();
  uint8_t maxLevel = stream.Read<uint8_t>();
  uint64_t expOffset = stream.Read<uint64_t>();
  uint32_t sExpOffset = stream.Read<uint32_t>();
  gold_ = stream.Read<uint64_t>();
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
    const auto item = parseGenericItem(stream, itemData);
    inventoryItemMap_.insert(std::pair<uint8_t, std::shared_ptr<storage::Item>>(slotNum, std::move(item)));
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
    const pk2::ref::Item &item = itemData.getItemById(refItemId);
      
    if (item.typeId1 == 3) {
      // ITEM_
      if (item.typeId2 == 1) { //TODO: Narrow filters for AvatarInventory
        // ITEM_CH
        // ITEM_EU
        // AVATAR_
        storage::ItemEquipment parsedItem;
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
    masteries_.emplace_back(id,level);
    hasNextMastery = stream.Read<uint8_t>();
  }

  uint8_t unknownByte1 = stream.Read<uint8_t>(); // "not a counter"

  //=====================================================================================
  //====================================== Skills =======================================
  //=====================================================================================

  uint8_t hasNextSkill = stream.Read<uint8_t>();
  while (hasNextSkill == 1) {
    uint32_t id = stream.Read<uint32_t>();
    bool enabled = stream.Read<uint8_t>();
    skills_.emplace_back(id, enabled);
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
  position_ = parsePosition(stream);
  uint16_t angle = stream.Read<uint16_t>();

  //=====================================================================================
  //===================================== Movement ======================================
  //=====================================================================================

  uint8_t hasDestination = stream.Read<uint8_t>();
  uint8_t movementType = stream.Read<uint8_t>();
  std::cout << "movementType: " << (int)movementType << '\n';

  if (hasDestination) {
    uint16_t destinationRegion = stream.Read<uint16_t>();

    if (position_.isDungeon()) {
      // Dungeon
      uint32_t destinationOffsetX = stream.Read<uint32_t>();
      uint32_t destinationOffsetY = stream.Read<uint32_t>();
      uint32_t destinationOffsetZ = stream.Read<uint32_t>();
    } else {
      // World
      uint16_t destinationX = stream.Read<uint16_t>();
      uint16_t destinationY = stream.Read<uint16_t>();
      uint16_t destinationZ = stream.Read<uint16_t>();
    }
  } else {
    packet::enums::AngleAction angleAction_ = static_cast<packet::enums::AngleAction>(stream.Read<uint8_t>());
    uint16_t angle = stream.Read<uint16_t>(); // Represents the new angle, character is looking at
  }

  //=====================================================================================
  //======================================= State =======================================
  //=====================================================================================

  lifeState_ = static_cast<enums::LifeState>(stream.Read<uint8_t>());
  uint8_t unkByte0 = stream.Read<uint8_t>();
  motionState_ = static_cast<enums::MotionState>(stream.Read<uint8_t>());
  std::cout << "Motion state is " << static_cast<int>(motionState_) << '\n';
  bodyState_ = static_cast<enums::BodyState>(stream.Read<uint8_t>());
  walkSpeed_ = stream.Read<float>();
  std::cout << "walkSpeed: " << walkSpeed_ << '\n';
  runSpeed_ = stream.Read<float>();
  std::cout << "runSpeed: " << runSpeed_ << '\n';
  hwanSpeed_ = stream.Read<float>();
  std::cout << "hwanSpeed: " << hwanSpeed_ << '\n';
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

uint32_t ParsedServerAgentCharacterData::refObjId() const {
  return refObjId_;
}

uint64_t ParsedServerAgentCharacterData::gold() const {
  return gold_;
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

const std::map<uint8_t, std::shared_ptr<storage::Item>>& ParsedServerAgentCharacterData::inventoryItemMap() const {
  return inventoryItemMap_;
}

const std::vector<structures::Mastery>& ParsedServerAgentCharacterData::masteries() const {
  return masteries_;
}

const std::vector<structures::Skill>& ParsedServerAgentCharacterData::skills() const {
  return skills_;
}

packet::structures::Position ParsedServerAgentCharacterData::position() const {
  return position_;
}

float ParsedServerAgentCharacterData::walkSpeed() const {
  return walkSpeed_;
}

float ParsedServerAgentCharacterData::runSpeed() const {
  return runSpeed_;
}

float ParsedServerAgentCharacterData::hwanSpeed() const {
  return hwanSpeed_;
}

enums::LifeState ParsedServerAgentCharacterData::lifeState() const {
  return lifeState_;
}

enums::MotionState ParsedServerAgentCharacterData::motionState() const {
  return motionState_;
}

enums::BodyState ParsedServerAgentCharacterData::bodyState() const {
  return bodyState_;
}

} // namespace packet::parsing