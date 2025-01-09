#include "serverAgentChatUpdate.hpp"

namespace packet::building {
  
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

PacketContainer ServerAgentChatUpdate::notice(const std::string &message) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(packet::enums::ChatType::kNotice));
  stream.Write<uint16_t>(message.size());
  stream.Write_Ascii(message);
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
  stream.Write<uint16_t>(message.size());
  stream.Write_Ascii(message);
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
    stream.Write<uint16_t>(senderName.size());
    stream.Write_Ascii(senderName);
  }
  stream.Write<uint16_t>(message.size());
  stream.Write_Ascii(message);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building