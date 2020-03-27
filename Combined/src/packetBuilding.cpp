#include "packetBuilding.hpp"

#include <iostream>
#include <limits>

namespace PacketBuilding {

namespace client_item_move {

PacketContainer base(const StreamUtility &stream) {
  return PacketContainer(static_cast<uint16_t>(Opcode::CLIENT_ITEM_MOVE), stream, false, false);
}

PacketContainer withinInventory(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(packet_enums::ItemMovementType::kWithinInventory));
  stream.Write<uint8_t>(srcSlot);
  stream.Write<uint8_t>(destSlot);
  stream.Write<uint16_t>(quantity);
  return base(stream);
}

} // namespace client_item_move

PacketBuilder::PacketBuilder(Opcode opcode) : opcode_(opcode) {}

PacketContainer PacketBuilder::packet(const StreamUtility &stream, bool encrypted, bool massive) const {
  return PacketContainer(static_cast<uint16_t>(opcode_), stream, (encrypted ? 1 : 0), (massive ? 1 : 0));
}

ClientAgentSelectionJoinPacketBuilder::ClientAgentSelectionJoinPacketBuilder(const std::string &name) :
      PacketBuilder(Opcode::CLIENT_INGAME_REQUEST),
      name_(name) {
  //
}

PacketContainer ClientAgentSelectionJoinPacketBuilder::packet() const {
  StreamUtility packetData;
  packetData.Write<uint16_t>(static_cast<uint16_t>(name_.size()));
  packetData.Write_Ascii(name_);
  return PacketBuilder::packet(packetData, true, false);
}

CharacterSelectionActionPacketBuilder::CharacterSelectionActionPacketBuilder(packet_enums::CharacterSelectionAction action) :
      PacketBuilder(Opcode::CLIENT_CHARACTER),
      action_(action) {
  //
}

PacketContainer CharacterSelectionActionPacketBuilder::packet() const {
  StreamUtility packetData;
  packetData.Write<uint8_t>(static_cast<uint8_t>(action_));
  return PacketBuilder::packet(packetData, true, false);

}

ClientAuthPacketBuilder::ClientAuthPacketBuilder(uint32_t loginToken, const std::string &kUsername, const std::string &kPassword, uint8_t kLocale, const std::array<uint8_t,6> &macAddress) :
      PacketBuilder(Opcode::CLIENT_AUTH),
      loginToken_(loginToken),
      kUsername_(kUsername),
      kPassword_(kPassword),
      kLocale_(kLocale),
      macAddress_(macAddress) {
  //
}

PacketContainer ClientAuthPacketBuilder::packet() const {
  StreamUtility clientAuthPacketData;
  clientAuthPacketData.Write<uint32_t>(loginToken_);
  clientAuthPacketData.Write<uint16_t>(kUsername_.size());
  clientAuthPacketData.Write_Ascii(kUsername_);
  clientAuthPacketData.Write<uint16_t>(kPassword_.size());
  clientAuthPacketData.Write_Ascii(kPassword_);
  clientAuthPacketData.Write<uint8_t>(kLocale_); //Content.ID
  for (const uint8_t macAddrByte : macAddress_) {
    clientAuthPacketData.Write<uint8_t>(macAddrByte);
  }
  return PacketBuilder::packet(clientAuthPacketData, true, false);
}

ClientCaptchaBuilder::ClientCaptchaBuilder(const std::string &answer) : PacketBuilder(Opcode::LOGIN_CLIENT_CAPTCHA_ANSWER), answer_(answer) {
  //
}

PacketContainer ClientCaptchaBuilder::packet() const {
  StreamUtility clientCaptchaPacketData;
  clientCaptchaPacketData.Write<uint16_t>(answer_.size());
  clientCaptchaPacketData.Write_Ascii(answer_);
  return PacketBuilder::packet(clientCaptchaPacketData, true, false);
}

ClientUseItemBuilder::ClientUseItemBuilder(uint8_t slotNum, uint16_t itemData) : PacketBuilder(Opcode::CLIENT_ITEM_USE), slotNum_(slotNum), itemData_(itemData) {
  //
}

PacketContainer ClientUseItemBuilder::packet() const {
  StreamUtility clientUseItemData;
  clientUseItemData.Write<uint8_t>(slotNum_);
  clientUseItemData.Write<uint16_t>(itemData_);

  /*
  //TID1 | TID2 | TID3 | TID4 | Name
  //3    | 3    | 1    | 4    | ITEM_ETC_POTION_COS_HP
  //3    | 3    | 1    | 9    | ITEM_ETC_POTION_COS_HGP
  //3    | 3    | 1    | 10   | ITEM_FORT_REPAIR_KIT
  //3    | 3    | 2    | 7    | ITEM_ETC_CURE_COS_ALL
  //3    | 3    | 6    | 2    | ITEM_FORT_SHOCK_BOMB
  clientUseItemData.Write<uint32_t>(targetGId);

  //3    | 3    | 1    | 6    | ITEM_ETC_POTION_COS_REVIVE
  //3    | 3    | 13   | 8    | ITEM_ETC_SPECIAL_EQUIP_TRANSGENDER
  //3    | 3    | 13   | 11   | ITEM_ETC_SPECIAL_EQUIP_REINFORCE
  //3    | 3    | 13   | 12   | ITEM_COS_P_EXTENSION
  //3    | 3    | 13   | 15   | ITEM_PET_HELPER
  //3    | 3    | 13   | 16   | ITEM_ETC_NASRUN_EXTENSION
  clientUseItemData.Write<uint8_t>(targetSlot);

  //3    | 3    | 3    | 3    | ITEM_ETC_SCROLL_REVERSE_RETURN
  clientUseItemData.Write<uint8_t>(reverseOption); // 2 = last recall point, 3 = last death location, 7 = location on map
  if (reverseOption == 7) {
    clientUseItemData.Write<uint32_t>(optionalTeleportId);
  }

  //3    | 3    | 3    | 5    | ITEM_ETC_SCROLL_CHATTING
  clientUseItemData.Write<uint16_t>(message.size());
  clientUseItemData.Write_Ascii(message);

  //3    | 3    | 3    | 6    | ITEM_FORT_TABLET
  //3    | 3    | 3    | 7    | ITEM_ETC_SCROLL_COS_GUARD
  //3    | 3    | 3    | 9    | ITEM_FORT_FLAG
  clientUseItemData.Write<uint32_t>(fortressID);
  clientUseItemData.Write<uint16_t>(rid);
  clientUseItemData.Write<uint32_t>(xOffset);
  clientUseItemData.Write<uint32_t>(yOffset);
  clientUseItemData.Write<uint32_t>(zOffset);

  //3    | 3    | 3    | 8    | ITEM_FORT_MANUAL
  clientUseItemData.Write<uint32_t>(fortressID);
  clientUseItemData.Write<uint32_t>(refEventID);

  //3    | 3    | 12   | 2    | ITEM_ETC_GUILD_CREST
  //NOTE: ONLY ON ITEM_ETC_GUILD_CREST_USER & ITEM_ETC_UNION_CREST_USER
  256 byte[]  imagePayload

  //3    | 3    | 13   | 9    | ITEM_ETC_SKIN_CHANGE
  clientUseItemData.Write<uint32_t>(refObjId);
  clientUseItemData.Write<uint8_t>(scale);
  */
  return PacketBuilder::packet(clientUseItemData, true, false);
}

LoginAuthPacketBuilder::LoginAuthPacketBuilder(uint8_t locale, const std::string &username, const std::string &password, uint16_t shardId) :
      PacketBuilder(Opcode::LOGIN_CLIENT_AUTH),
      locale_(locale),
      username_(username),
      password_(password),
      shardId_(shardId) {
  //
}

PacketContainer LoginAuthPacketBuilder::packet() const {
  StreamUtility loginAuthPacketData;
  loginAuthPacketData.Write<uint8_t>(locale_);
  loginAuthPacketData.Write<uint16_t>(username_.size());
  loginAuthPacketData.Write_Ascii(username_);
  loginAuthPacketData.Write<uint16_t>(password_.size());
  loginAuthPacketData.Write_Ascii(password_);
  loginAuthPacketData.Write<uint16_t>(shardId_);
  return PacketBuilder::packet(loginAuthPacketData, true, false);
}

ClientMovementRequestPacketBuilder::ClientMovementRequestPacketBuilder(uint16_t angle) :
      PacketBuilder(Opcode::CLIENT_MOVEMENT),
      hasDestination_(false),
      angle_(angle) {
  //
}

ClientMovementRequestPacketBuilder::ClientMovementRequestPacketBuilder(uint16_t regionId, uint32_t xOffset, uint32_t yOffset, uint32_t zOffset) :
      PacketBuilder(Opcode::CLIENT_MOVEMENT),
      hasDestination_(true),
      regionId_(regionId),
      xOffset_(xOffset),
      yOffset_(yOffset),
      zOffset_(zOffset) {
  //
}

PacketContainer ClientMovementRequestPacketBuilder::packet() const {
  // https://www.elitepvpers.com/forum/sro-coding-corner/1992345-coordinate-converter-open-source.html#post17651107
  // 1   bool    HasDestination
  // if(HasDestination)
  // {
  //     2   ushort  Destination.RegionID
  //     if(Destination.RegionID < short.MaxValue)
  //     {
  //         //World
  //         2   ushort  Destination.XOffset
  //         2   ushort  Destination.YOffset
  //         2   ushort  Destination.ZOffset
  //     }
  //     else
  //     {
  //         //Dungeon
  //         4   uint  Destination.XOffset
  //         4   uint  Destination.YOffset
  //         4   uint  Destination.ZOffset
  //     }
  // }
  // else
  // {
  //     1   byte    AngleAction
  //     2   ushort  Angle
  // }

  // public enum AngleAction : byte
  // {
  //     Obsolete = 0, //GO_BACKWARDS or SPIN?
  //     GoForward = 1
  // }
  StreamUtility clientMovementRequestPacketData;
  clientMovementRequestPacketData.Write<uint8_t>(hasDestination_ ? 1 : 0);
  if (hasDestination_) {
    clientMovementRequestPacketData.Write<uint16_t>(regionId_);
    if (regionId_ < std::numeric_limits<int16_t>::max()) {
      //World
      clientMovementRequestPacketData.Write<uint16_t>(xOffset_);
      clientMovementRequestPacketData.Write<uint16_t>(yOffset_);
      clientMovementRequestPacketData.Write<uint16_t>(zOffset_);
    } else {
      //Dungeon
      clientMovementRequestPacketData.Write<uint32_t>(xOffset_);
      clientMovementRequestPacketData.Write<uint32_t>(yOffset_);
      clientMovementRequestPacketData.Write<uint32_t>(zOffset_);
    }
  } else {
    clientMovementRequestPacketData.Write<uint8_t>(static_cast<uint8_t>(packet_enums::AngleAction::kGoForward));
    clientMovementRequestPacketData.Write<uint16_t>(angle_);
  }
  return PacketBuilder::packet(clientMovementRequestPacketData, true, false);
}

ServerChatPacketBuilder::ServerChatPacketBuilder(packet_enums::ChatType chatType, const std::string &message) :
      PacketBuilder(Opcode::SERVER_CHAT),
      chatType_(chatType),
      message_(message) {
  //
}

ServerChatPacketBuilder::ServerChatPacketBuilder(packet_enums::ChatType chatType, uint32_t senderId, const std::string &message) :
      PacketBuilder(Opcode::SERVER_CHAT),
      chatType_(chatType),
      senderId_(senderId),
      message_(message) {
  //
}

ServerChatPacketBuilder::ServerChatPacketBuilder(packet_enums::ChatType chatType, const std::string &senderName, const std::string &message) :
      PacketBuilder(Opcode::SERVER_CHAT),
      chatType_(chatType),
      senderName_(senderName),
      message_(message) {
  //
  
}

PacketContainer ServerChatPacketBuilder::packet() const {
  // Inject a server global notice with this command
  // 1   byte    chatType
  // if(chatType == ChatType.All ||
  //   chatType == ChatType.AllGM ||
  //   chatType == ChatType.NPC)
  // {
  //     4   uint    message.Sender.UniqueID
  // }
  // else if(chatType == ChatType.PM ||
  //         chatType == ChatType.Party ||
  //         chatType == ChatType.Guild ||
  //         chatType == ChatType.Global ||
  //         chatType == ChatType.Stall ||
  //         chatType == ChatType.Union ||
  //         chatType == ChatType.Accademy)        
  // {
  //     2   ushort  message.Sender.Name.Length
  //     *   string  message.Sender.Name
  // }
  // 2   ushort  message.Length
  // *   string  message
  StreamUtility serverChatPacketData;
  serverChatPacketData.Write<uint8_t>(static_cast<uint8_t>(chatType_));
  if (chatType_ == packet_enums::ChatType::kAll ||
      chatType_ == packet_enums::ChatType::kAllGm ||
      chatType_ == packet_enums::ChatType::kNpc) {
    serverChatPacketData.Write<uint32_t>(senderId_);
  } else if (chatType_ == packet_enums::ChatType::kPm ||
             chatType_ == packet_enums::ChatType::kParty ||
             chatType_ == packet_enums::ChatType::kGuild ||
             chatType_ == packet_enums::ChatType::kGlobal ||
             chatType_ == packet_enums::ChatType::kStall ||
             chatType_ == packet_enums::ChatType::kUnion ||
             chatType_ == packet_enums::ChatType::kAcademy) {
    serverChatPacketData.Write<uint16_t>(senderName_.size());
    serverChatPacketData.Write_Ascii(senderName_);
  }
  serverChatPacketData.Write<uint16_t>(message_.size());
  serverChatPacketData.Write_Ascii(message_);
  return PacketBuilder::packet(serverChatPacketData, true, false);
}


} // namespace PacketBuilding