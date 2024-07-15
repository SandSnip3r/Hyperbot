#ifndef PACKET_PARSING_SERVER_AGENT_CHARACTER_INCREASE_INT_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_AGENT_CHARACTER_INCREASE_INT_RESPONSE_HPP_

#include "packet/parsing/parsedPacket.hpp"

namespace packet::parsing {

class ServerAgentCharacterIncreaseIntResponse : public ParsedPacket {
public:
  ServerAgentCharacterIncreaseIntResponse(const PacketContainer &packet);
  uint8_t result() const { return result_; }
  uint16_t errorCode() const { return errorCode_; }
private:
  uint8_t result_;
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_CHARACTER_INCREASE_INT_RESPONSE_HPP_