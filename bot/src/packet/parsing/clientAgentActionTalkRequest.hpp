#ifndef PACKET_PARSING_CLIENT_AGENT_ACTION_TALK_REQUEST_HPP_
#define PACKET_PARSING_CLIENT_AGENT_ACTION_TALK_REQUEST_HPP_

#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"

#include <cstdint>

namespace packet::parsing {
  
class ClientAgentActionTalkRequest : public ParsedPacket {
public:
  ClientAgentActionTalkRequest(const PacketContainer &packet);
  uint32_t gId() const;
  enums::TalkOption talkOption() const;
private:
  uint32_t gId_;
  enums::TalkOption talkOption_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_CLIENT_AGENT_ACTION_TALK_REQUEST_HPP_