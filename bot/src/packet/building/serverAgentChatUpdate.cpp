#include "serverAgentChatUpdate.hpp"

namespace packet::building {

PacketContainer ServerAgentChatUpdate::notice(const std::string &message) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(packet::enums::ChatType::kNotice));
  stream.Write(message);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ServerAgentChatUpdate::packet(packet::enums::ChatType chatType, uint32_t senderId, const std::string &message) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(chatType));
  if (chatType == packet::enums::ChatType::kAll ||
      chatType == packet::enums::ChatType::kAllGm ||
      chatType == packet::enums::ChatType::kNpc) {
    stream.Write<uint32_t>(senderId);
  }
  stream.Write(message);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ServerAgentChatUpdate::packet(packet::enums::ChatType chatType, const std::string &senderName, const std::string &message) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(chatType));
  if (chatType == packet::enums::ChatType::kPm ||
      chatType == packet::enums::ChatType::kParty ||
      chatType == packet::enums::ChatType::kGuild ||
      chatType == packet::enums::ChatType::kGlobal ||
      chatType == packet::enums::ChatType::kStall ||
      chatType == packet::enums::ChatType::kUnion ||
      chatType == packet::enums::ChatType::kAcademy) {
    stream.Write(senderName);
  }
  stream.Write(message);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building