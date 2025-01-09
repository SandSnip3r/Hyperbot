#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_MOVE_SPEED_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_MOVE_SPEED_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"

#include <silkroad_lib/scalar_types.h>

#include <cstdint>
#include <string>

namespace packet::parsing {
  
class ServerAgentEntityUpdateMoveSpeed : public ParsedPacket {
public:
  ServerAgentEntityUpdateMoveSpeed(const PacketContainer &packet);
  sro::scalar_types::EntityGlobalId globalId() const;
  float walkSpeed() const;
  float runSpeed() const;
private:
  sro::scalar_types::EntityGlobalId globalId_;
  float walkSpeed_;
  float runSpeed_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_MOVE_SPEED_HPP