#ifndef PACKET_PARSING_SERVER_AGENT_CHAT_UPDATE_HPP
#define PACKET_PARSING_SERVER_AGENT_CHAT_UPDATE_HPP

#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"

#include <cstdint>
#include <string>

namespace packet::parsing {

class ServerAgentChatUpdate : public ParsedPacket {
public:
  ServerAgentChatUpdate(const PacketContainer &packet);
  packet::enums::ChatType chatType() const;
  uint32_t senderGlobalId() const;
  std::string senderName() const;
  std::string message() const;
private:
  packet::enums::ChatType chatType_;
  uint32_t senderGlobalId_;
  std::string senderName_;
  std::string message_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_CHAT_UPDATE_HPP