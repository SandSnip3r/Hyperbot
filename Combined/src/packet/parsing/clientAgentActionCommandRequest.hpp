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
  structures::ActionCommand actionCommand() const;
  enums::CommandType commandType() const;
  enums::ActionType actionType() const;
  uint32_t refSkillId() const;
  enums::TargetType targetType() const;
  uint32_t targetGlobalId() const;
  uint16_t regionId() const;
  float x() const;
  float y() const;
  float z() const;
private:
  structures::ActionCommand actionCommand_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_CLIENT_AGENT_ACTION_COMMAND_REQUEST_HPP