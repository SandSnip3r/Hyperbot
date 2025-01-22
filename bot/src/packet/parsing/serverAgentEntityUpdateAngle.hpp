#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_ANGLE_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_ANGLE_HPP

#include "parsedPacket.hpp"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/scalar_types.hpp>

namespace packet::parsing {

class ServerAgentEntityUpdateAngle : public ParsedPacket {
public:
  ServerAgentEntityUpdateAngle(const PacketContainer &packet);
  sro::scalar_types::EntityGlobalId globalId() const;
  sro::Angle angle() const;
private:
  sro::scalar_types::EntityGlobalId globalId_;
  sro::Angle angle_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_ANGLE_HPP