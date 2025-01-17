#ifndef PACKET_PARSING_CLIENT_AGENT_CHAT_REQUEST_HPP
#define PACKET_PARSING_CLIENT_AGENT_CHAT_REQUEST_HPP

#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"

#include <cstdint>
#include <string>

namespace packet::parsing {

class ClientAgentChatRequest : public ParsedPacket {
public:
  ClientAgentChatRequest(const PacketContainer &packet);
  packet::enums::ChatType chatType() const;
  uint8_t chatIndex() const;
  const std::string& receiverName() const;
  const std::string& message() const;
private:
  packet::enums::ChatType chatType_;
  uint8_t chatIndex_;
  std::string receiverName_;
  std::string message_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_CLIENT_AGENT_CHAT_REQUEST_HPP