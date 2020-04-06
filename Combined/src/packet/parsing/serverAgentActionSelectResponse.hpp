#ifndef PACKET_PARSING_SEVER_AGENT_ACTION_SELECT_RESPONSE_HPP
#define PACKET_PARSING_SEVER_AGENT_ACTION_SELECT_RESPONSE_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"

// #include <cstdint>
// #include <string>

namespace packet::parsing {
  
class ServerAgentActionSelectResponse : public ParsedPacket {
public:
  ServerAgentActionSelectResponse(const PacketContainer &packet);
  uint8_t result() const;
  uint32_t gId() const;
  uint16_t errorCode() const;
private:
  uint8_t result_;
  uint32_t gId_;
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SEVER_AGENT_ACTION_SELECT_RESPONSE_HPP