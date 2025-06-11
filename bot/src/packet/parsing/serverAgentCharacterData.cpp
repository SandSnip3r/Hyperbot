#include "commonParsing.hpp"
#include "serverAgentCharacterData.hpp"

namespace packet::parsing {

ServerAgentCharacterData::ServerAgentCharacterData(const PacketContainer &packet, const sro::pk2::ItemData &itemData, const sro::pk2::SkillData &skillData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint32_t serverTime = stream.Read<uint32_t>();
  stream.Read(refObjId_);
  uint8_t scale = stream.Read<uint8_t>();
  curLevel_ = stream.Read<uint8_t>();
  uint8_t maxLevel = stream.Read<uint8_t>();
  currentExperience_ = stream.Read<uint64_t>();
  currentSpExperience_ = stream.Read<uint32_t>();
  gold_ = stream.Read<uint64_t>();
  skillPoints_ = stream.Read<uint32_t>();
  stream.Read(availableStatPoints_);
  stream.Read(hwanPoints_);
  uint32_t gatheredExpPoint = stream.Read<uint32_t>();
  hp_ = stream.Read<uint32_t>();
  mp_ = stream.Read<uint32_t>();
  uint8_t autoInverstExp = stream.Read<uint8_t>();
  uint8_t dailyPK = stream.Read<uint8_t>();
  uint16_t totalPK = stream.Read<uint16_t>();
  uint32_t pKPenaltyPoint = stream.Read<uint32_t>();
  stream.Read(hwanLevel_);
  uint8_t freePVP = stream.Read<uint8_t>(); // 0 = None, 1 = Red, 2 = Gray, 3 = Blue, 4 = White, 5 = Gold

  //=====================================================================================
  //===================================== Inventory =====================================
  //=====================================================================================

  inventorySize_ = stream.Read<uint8_t>();

  uint8_t inventoryItemCount = stream.Read<uint8_t>();

  for (int itemNum=0; itemNum<inventoryItemCount; ++itemNum) {
    uint8_t slotNum = stream.Read<uint8_t>();
    auto item = parseGenericItem(stream, itemData);
    inventoryItemMap_.insert(std::pair<uint8_t, std::shared_ptr<storage::Item>>(slotNum, std::move(item)));
  }

  //=====================================================================================
  //================================== Avatar Inventory =================================
  //=====================================================================================

  avatarInventorySize_ = stream.Read<uint8_t>();
  uint8_t avatarItemCount = stream.Read<uint8_t>();

  for (int i=0; i<avatarItemCount; ++i) {
    uint8_t slotNum = stream.Read<uint8_t>();
    auto item = parseGenericItem(stream, itemData);
    avatarInventoryItemMap_.insert(std::pair<uint8_t, std::shared_ptr<storage::Item>>(slotNum, std::move(item)));
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
  
  stream.Read(globalId_);

  //=====================================================================================
  //===================================== Position ======================================
  //=====================================================================================
  position_ = parsePosition(stream);
  angle_ = stream.Read<sro::Angle>();

  //=====================================================================================
  //===================================== Movement ======================================
  //=====================================================================================

  uint8_t hasDestination = stream.Read<uint8_t>();
  uint8_t movementType = stream.Read<uint8_t>();

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

  lifeState_ = stream.Read<sro::entity::LifeState>();
  uint8_t unkByte0 = stream.Read<uint8_t>();
  motionState_ = stream.Read<entity::MotionState>();
  bodyState_ = static_cast<enums::BodyState>(stream.Read<uint8_t>());
  walkSpeed_ = stream.Read<float>();
  runSpeed_ = stream.Read<float>();
  hwanSpeed_ = stream.Read<float>();
  uint8_t buffCount = stream.Read<uint8_t>();
  if (buffCount > 0) {
    LOG(INFO) << "Spawned with " << buffCount << " buffs!";
  }
  for (int i=0; i<buffCount; ++i) {
    // We seem to always spawn with 0 buffs
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
  characterName_ = stream.Read_Ascii(nameLength);
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
  jId_ = stream.Read<uint32_t>();
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

uint8_t ServerAgentCharacterData::curLevel() const {
  return curLevel_;
}

uint64_t ServerAgentCharacterData::currentExperience() const {
  return currentExperience_;
}

uint32_t ServerAgentCharacterData::currentSpExperience() const {
  return currentSpExperience_;
}

uint64_t ServerAgentCharacterData::gold() const {
  return gold_;
}

uint32_t ServerAgentCharacterData::skillPoints() const {
  return skillPoints_;
}

uint16_t ServerAgentCharacterData::availableStatPoints() const {
  return availableStatPoints_;
}

uint8_t ServerAgentCharacterData::hwanPoints() const {
  return hwanPoints_;
}

uint32_t ServerAgentCharacterData::hp() const {
  return hp_;
}

uint32_t ServerAgentCharacterData::mp() const {
  return mp_;
}

uint8_t ServerAgentCharacterData::hwanLevel() const {
  return hwanLevel_;
}

uint8_t ServerAgentCharacterData::inventorySize() const {
  return inventorySize_;
}

const std::map<uint8_t, std::shared_ptr<storage::Item>>& ServerAgentCharacterData::inventoryItemMap() const {
  return inventoryItemMap_;
}

uint8_t ServerAgentCharacterData::avatarInventorySize() const {
  return avatarInventorySize_;
}

const std::map<uint8_t, std::shared_ptr<storage::Item>>& ServerAgentCharacterData::avatarInventoryItemMap() const {
  return avatarInventoryItemMap_;
}

const std::vector<structures::Mastery>& ServerAgentCharacterData::masteries() const {
  return masteries_;
}

const std::vector<structures::Skill>& ServerAgentCharacterData::skills() const {
  return skills_;
}

sro::Position ServerAgentCharacterData::position() const {
  return position_;
}

sro::Angle ServerAgentCharacterData::angle() const {
  return angle_;
}

float ServerAgentCharacterData::walkSpeed() const {
  return walkSpeed_;
}

float ServerAgentCharacterData::runSpeed() const {
  return runSpeed_;
}

float ServerAgentCharacterData::hwanSpeed() const {
  return hwanSpeed_;
}

std::string ServerAgentCharacterData::characterName() const {
  return characterName_;
}

sro::entity::LifeState ServerAgentCharacterData::lifeState() const {
  return lifeState_;
}

entity::MotionState ServerAgentCharacterData::motionState() const {
  return motionState_;
}

enums::BodyState ServerAgentCharacterData::bodyState() const {
  return bodyState_;
}

uint32_t ServerAgentCharacterData::jId() const {
  return jId_;
}

} // namespace packet::parsing