#include "commonParsing.hpp"
#include "parsedPacket.hpp"

#include <silkroad_lib/position_math.h>

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
  if (movement_.type == packet::enums::ItemMovementType::kUpdateSlotsInventory) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kUpdateSlotsChest ||
             movement_.type == packet::enums::ItemMovementType::kUpdateSlotsGuildChest) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
    uint32_t unk0 = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kChestDepositItem ||
             movement_.type == packet::enums::ItemMovementType::kChestWithdrawItem ||
             movement_.type == packet::enums::ItemMovementType::kGuildChestDepositItem ||
             movement_.type == packet::enums::ItemMovementType::kGuildChestWithdrawItem) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint32_t unk0 = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kBuyItem) {
    movement_.storeTabNumber = stream.Read<uint8_t>();
    movement_.storeSlotNumber = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
    movement_.globalId = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kSellItem) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
    uint32_t unk1 = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kDropGold ||
             movement_.type == packet::enums::ItemMovementType::kChestWithdrawGold ||
             movement_.type == packet::enums::ItemMovementType::kChestDepositGold ||
             movement_.type == packet::enums::ItemMovementType::kGuildChestDepositGold ||
             movement_.type == packet::enums::ItemMovementType::kGuildChestWithdrawGold) {
    uint64_t goldAmount = stream.Read<uint64_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kMoveItemCosToInventory ||
             movement_.type == packet::enums::ItemMovementType::kMoveItemInventoryToCos) {
    uint32_t unk4 = stream.Read<uint32_t>();
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kUpdateSlotsInventoryCos) {
    uint32_t unk4 = stream.Read<uint32_t>();
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kMoveItemAvatarToInventory) {
    uint8_t sourceAvatarInventorySlot = stream.Read<uint8_t>();
    uint8_t destInventorySlot = stream.Read<uint8_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kMoveItemInventoryToAvatar) {
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

//=========================================================================================================================================================

} // namespace packet::parsing