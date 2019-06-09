#include "opcode.hpp"
#include "packetParsing.hpp"

namespace PacketParsing {

PacketParser* newPacketParser(const PacketContainer &packet) {
  // Given a packet's opcode, determine which parser is appropriate
  switch (static_cast<Opcode>(packet.opcode)) {
    case Opcode::CLIENT_CHAT:
      return new ClientChatPacket(packet);
  }
  return nullptr;
}

void PacketParser::parsedCheck() {
  // Lazy-eval mechanism
  if (!parsed_) {
    parsePacket();
    parsed_ = true;
  }
}

ClientChatPacket::ClientChatPacket(const PacketContainer &packet) : packet_(packet) {}

PacketEnums::ChatType ClientChatPacket::chatType() {
  parsedCheck();
  return chatType_;
}

uint8_t ClientChatPacket::chatIndex() {
  parsedCheck();
  return chatIndex_;
}

const std::string& ClientChatPacket::receiverName() {
  parsedCheck();
  return receiverName_;
}

const std::string& ClientChatPacket::message() {
  parsedCheck();
  return message_;
}

void ClientChatPacket::parsePacket() {
  // 1   byte    chatType
  // 1   byte    chatIndex
  // if(chatType == ChatType.PM)
  // {
  //     2   ushort  reciver.Length
  //     *   string  reciver
  // }
  // 2   ushort  message.Length
  // *   string  message
  StreamUtility stream = packet_.data;
  chatType_ = static_cast<PacketEnums::ChatType>(stream.Read<uint8_t>());
  chatIndex_ = stream.Read<uint8_t>();
  if (chatType_ == PacketEnums::ChatType::kPm) {
    const uint16_t kReceiverNameLength = stream.Read<uint16_t>();
    receiverName_ = stream.Read_Ascii(kReceiverNameLength);
  }
  const uint16_t kMessageLength = stream.Read<uint16_t>();
  message_ = stream.Read_Ascii(kMessageLength);
}

} // namespace PacketParsing