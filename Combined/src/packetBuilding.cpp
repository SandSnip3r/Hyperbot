#include "packetBuilding.hpp"

#include <iostream>
#include <limits>

namespace PacketBuilding {

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

CharacterSelectionActionPacketBuilder::CharacterSelectionActionPacketBuilder(PacketEnums::CharacterSelectionAction action) :
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
    clientMovementRequestPacketData.Write<uint8_t>(static_cast<uint8_t>(PacketEnums::AngleAction::kGoForward));
    clientMovementRequestPacketData.Write<uint16_t>(angle_);
  }
  return PacketBuilder::packet(clientMovementRequestPacketData, true, false);
}

ServerChatPacketBuilder::ServerChatPacketBuilder(PacketEnums::ChatType chatType, const std::string &message) :
      PacketBuilder(Opcode::SERVER_CHAT),
      chatType_(chatType),
      message_(message) {
  //
}

ServerChatPacketBuilder::ServerChatPacketBuilder(PacketEnums::ChatType chatType, uint32_t senderId, const std::string &message) :
      PacketBuilder(Opcode::SERVER_CHAT),
      chatType_(chatType),
      senderId_(senderId),
      message_(message) {
  //
}

ServerChatPacketBuilder::ServerChatPacketBuilder(PacketEnums::ChatType chatType, const std::string &senderName, const std::string &message) :
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
  if (chatType_ == PacketEnums::ChatType::kAll ||
      chatType_ == PacketEnums::ChatType::kAllGm ||
      chatType_ == PacketEnums::ChatType::kNpc) {
    serverChatPacketData.Write<uint32_t>(senderId_);
  } else if (chatType_ == PacketEnums::ChatType::kPm ||
             chatType_ == PacketEnums::ChatType::kParty ||
             chatType_ == PacketEnums::ChatType::kGuild ||
             chatType_ == PacketEnums::ChatType::kGlobal ||
             chatType_ == PacketEnums::ChatType::kStall ||
             chatType_ == PacketEnums::ChatType::kUnion ||
             chatType_ == PacketEnums::ChatType::kAcademy) {
    serverChatPacketData.Write<uint16_t>(senderName_.size());
    serverChatPacketData.Write_Ascii(senderName_);
  }
  serverChatPacketData.Write<uint16_t>(message_.size());
  serverChatPacketData.Write_Ascii(message_);
  return PacketBuilder::packet(serverChatPacketData, true, false);
}


} // namespace PacketBuilding