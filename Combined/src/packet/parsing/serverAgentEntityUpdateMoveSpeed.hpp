#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_MOVE_SPEED_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_MOVE_SPEED_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"

#include <cstdint>
#include <string>

namespace packet::parsing {
  
class ServerAgentEntityUpdateMoveSpeed : public ParsedPacket {
public:
  ServerAgentEntityUpdateMoveSpeed(const PacketContainer &packet);
  uint32_t globalId() const;
  float walkSpeed() const;
  float runSpeed() const;
private:
  uint32_t globalId_;
  float walkSpeed_;
  float runSpeed_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_MOVE_SPEED_HPP