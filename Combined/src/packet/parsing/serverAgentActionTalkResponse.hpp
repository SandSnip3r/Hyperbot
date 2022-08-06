#ifndef PACKET_PARSING_SEVER_AGENT_ACTION_TALK_RESPONSE_HPP
#define PACKET_PARSING_SEVER_AGENT_ACTION_TALK_RESPONSE_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"

#include <cstdint>

namespace packet::parsing {
  
class ServerAgentActionTalkResponse : public ParsedPacket {
public:
  ServerAgentActionTalkResponse(const PacketContainer &packet);
  uint8_t result() const;

  packet::enums::TalkOption talkOption() const;
  uint8_t isSpecialtyTime() const;
  
  uint16_t errorCode() const;
private:
  uint8_t result_;
  // Success case
  packet::enums::TalkOption talkOption_;
  uint8_t isSpecialtyTime_;
  // Error case
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SEVER_AGENT_ACTION_TALK_RESPONSE_HPP