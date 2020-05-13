#include "serverAgentChatUpdate.hpp"

namespace packet::parsing {

ServerAgentChatUpdate::ServerAgentChatUpdate(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  chatType_ = static_cast<packet::enums::ChatType>(stream.Read<uint8_t>());
  if (chatType_ == enums::ChatType::kAll ||
      chatType_ == enums::ChatType::kAllGm ||
      chatType_ == enums::ChatType::kNpc) {
    senderGlobalId_ = stream.Read<uint32_t>();
  } else if (chatType_ == enums::ChatType::kPm ||
             chatType_ == enums::ChatType::kParty ||
             chatType_ == enums::ChatType::kGuild ||
             chatType_ == enums::ChatType::kGlobal ||
             chatType_ == enums::ChatType::kStall ||
             chatType_ == enums::ChatType::kUnion ||
             chatType_ == enums::ChatType::kAcademy) {
    const uint16_t nameLength = stream.Read<uint16_t>();
    senderName_ = stream.Read_Ascii(nameLength);
  }
  const uint16_t kMessageLength = stream.Read<uint16_t>();
  message_ = stream.Read_Ascii(kMessageLength);
}

packet::enums::ChatType ServerAgentChatUpdate::chatType() const {
  return chatType_;
}

uint32_t ServerAgentChatUpdate::senderGlobalId() const {
  return senderGlobalId_;
}

std::string ServerAgentChatUpdate::senderName() const {
  return senderName_;
}

std::string ServerAgentChatUpdate::message() const {
  return message_;
}

} // namespace packet::parsing