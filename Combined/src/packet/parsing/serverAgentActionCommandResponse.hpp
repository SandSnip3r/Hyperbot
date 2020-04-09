#ifndef PACKET_PARSING_SERVER_AGENT_ACTION_COMMAND_RESPONSE_HPP
#define PACKET_PARSING_SERVER_AGENT_ACTION_COMMAND_RESPONSE_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"

#include <cstdint>

namespace packet::parsing {

enum ActionState : uint8_t {
  kBegin = 1,
  kEnd = 2,
  kError = 3
};
  
class ServerAgentActionCommandResponse : public ParsedPacket {
public:
  ServerAgentActionCommandResponse(const PacketContainer &packet);
  ActionState actionState() const;
  bool repeatAction() const;
  uint16_t errorCode() const;
private:
  ActionState actionState_;
  bool repeatAction_;
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ACTION_COMMAND_RESPONSE_HPP