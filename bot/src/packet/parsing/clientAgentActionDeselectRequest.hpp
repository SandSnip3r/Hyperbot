#ifndef PACKET_PARSING_CLIENT_AGENT_ACTION_DESELECT_REQUEST_HPP_
#define PACKET_PARSING_CLIENT_AGENT_ACTION_DESELECT_REQUEST_HPP_

#include "parsedPacket.hpp"

#include <cstdint>

namespace packet::parsing {
  
class ClientAgentActionDeselectRequest : public ParsedPacket {
public:
  ClientAgentActionDeselectRequest(const PacketContainer &packet);
  uint32_t gId() const;
private:
  uint32_t gId_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_CLIENT_AGENT_ACTION_DESELECT_REQUEST_HPP_