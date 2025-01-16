#ifndef PACKET_PARSING_SERVER_AGENT_AUTH_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_AGENT_AUTH_RESPONSE_HPP_

#include "packet/parsing/parsedPacket.hpp"

namespace packet::parsing {

class ServerAgentAuthResponse : public ParsedPacket {
public:
  ServerAgentAuthResponse(const PacketContainer &packet);
  uint8_t result() const { return result_; }
  uint8_t errorCode() const { return errorCode_; }
private:
  uint8_t result_;
  uint8_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_AUTH_RESPONSE_HPP_