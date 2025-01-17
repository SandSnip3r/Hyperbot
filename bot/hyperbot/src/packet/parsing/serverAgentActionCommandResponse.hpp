#ifndef PACKET_PARSING_SERVER_AGENT_ACTION_COMMAND_RESPONSE_HPP
#define PACKET_PARSING_SERVER_AGENT_ACTION_COMMAND_RESPONSE_HPP

#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"

#include <cstdint>

namespace packet::parsing {

class ServerAgentActionCommandResponse : public ParsedPacket {
public:
  ServerAgentActionCommandResponse(const PacketContainer &packet);
  enums::ActionState actionState() const;
  bool repeatAction() const;
  uint16_t errorCode() const;
private:
  enums::ActionState actionState_;
  bool repeatAction_;
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ACTION_COMMAND_RESPONSE_HPP