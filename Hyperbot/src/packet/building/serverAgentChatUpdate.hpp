#include "../enums/packetEnums.hpp"
#include "../opcode.hpp"
#include "../../shared/silkroad_security.h"

#ifndef PACKET_BUILDING_SERVER_AGENT_CHAT_UPDATE_HPP
#define PACKET_BUILDING_SERVER_AGENT_CHAT_UPDATE_HPP

namespace packet::building {

class ServerAgentChatUpdate {
private:
  static const Opcode kOpcode_ = Opcode::kServerAgentChatUpdate;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer notice(const std::string &message);
  static PacketContainer packet(packet::enums::ChatType chatType, uint32_t senderId, const std::string &message);
  static PacketContainer packet(packet::enums::ChatType chatType, const std::string &senderName, const std::string &message);
};

} // namespace packet::building

#endif // PACKET_BUILDING_SERVER_AGENT_CHAT_UPDATE_HPP