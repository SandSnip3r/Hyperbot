#ifndef PACKET_PARSING_CLIENT_AGENT_ACTION_SELECT_REQUEST_HPP_
#define PACKET_PARSING_CLIENT_AGENT_ACTION_SELECT_REQUEST_HPP_

#include "parsedPacket.hpp"

#include <cstdint>

namespace packet::parsing {
  
class ClientAgentActionSelectRequest : public ParsedPacket {
public:
  ClientAgentActionSelectRequest(const PacketContainer &packet);
  uint32_t gId() const;
private:
  uint32_t gId_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_CLIENT_AGENT_ACTION_SELECT_REQUEST_HPP_