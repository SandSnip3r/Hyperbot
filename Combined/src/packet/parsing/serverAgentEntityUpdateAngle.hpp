#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_ANGLE_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_ANGLE_HPP

#include "parsedPacket.hpp"

#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>

namespace packet::parsing {
  
class ServerAgentEntityUpdateAngle : public ParsedPacket {
public:
  ServerAgentEntityUpdateAngle(const PacketContainer &packet);
  sro::scalar_types::EntityGlobalId globalId() const;
  sro::MovementAngle angle() const;
private:
  sro::scalar_types::EntityGlobalId globalId_;
  sro::MovementAngle angle_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_ANGLE_HPP