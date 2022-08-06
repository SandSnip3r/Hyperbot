#ifndef PACKET_PARSING_SEVER_AGENT_ACTION_DESELECT_RESPONSE_HPP_
#define PACKET_PARSING_SEVER_AGENT_ACTION_DESELECT_RESPONSE_HPP_

#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"

#include <cstdint>

namespace packet::parsing {
  
class ServerAgentActionDeselectResponse : public ParsedPacket {
public:
  ServerAgentActionDeselectResponse(const PacketContainer &packet);
  uint8_t result() const;
  uint16_t errorCode() const;
private:
  uint8_t result_;
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SEVER_AGENT_ACTION_DESELECT_RESPONSE_HPP_