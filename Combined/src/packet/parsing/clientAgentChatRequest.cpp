#include "clientAgentChatRequest.hpp"

namespace packet::parsing {

ClientAgentChatRequest::ClientAgentChatRequest(const PacketContainer &packet) :
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
  chatType_ = static_cast<packet::enums::ChatType>(stream.Read<uint8_t>());
  chatIndex_ = stream.Read<uint8_t>();
  if (chatType_ == packet::enums::ChatType::kPm) {
    const uint16_t kReceiverNameLength = stream.Read<uint16_t>();
    receiverName_ = stream.Read_Ascii(kReceiverNameLength);
  }
  const uint16_t kMessageLength = stream.Read<uint16_t>();
  message_ = stream.Read_Ascii(kMessageLength);
}

packet::enums::ChatType ClientAgentChatRequest::chatType() const {
  return chatType_;
}

uint8_t ClientAgentChatRequest::chatIndex() const {
  return chatIndex_;
}

const std::string& ClientAgentChatRequest::receiverName() const {
  return receiverName_;
}

const std::string& ClientAgentChatRequest::message() const {
  return message_;
}

} // namespace packet::parsing