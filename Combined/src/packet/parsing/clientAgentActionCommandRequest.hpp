#ifndef PACKET_PARSING_CLIENT_AGENT_ACTION_COMMAND_REQUEST_HPP
#define PACKET_PARSING_CLIENT_AGENT_ACTION_COMMAND_REQUEST_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"
#include "../structures/packetInnerStructures.hpp"

#include <cstdint>

namespace packet::parsing {
  
class ClientAgentActionCommandRequest : public ParsedPacket {
public:
  ClientAgentActionCommandRequest(const PacketContainer &packet);
  const structures::ActionCommand& actionCommand() const;
private:
  structures::ActionCommand actionCommand_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_CLIENT_AGENT_ACTION_COMMAND_REQUEST_HPP