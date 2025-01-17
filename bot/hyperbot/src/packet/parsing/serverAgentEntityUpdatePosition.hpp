#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POSITION_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POSITION_HPP

#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include <silkroad_lib/position.h>

#include <cstdint>

namespace packet::parsing {

class ServerAgentEntityUpdatePosition : public ParsedPacket {
public:
  ServerAgentEntityUpdatePosition(const PacketContainer &packet);
  uint32_t globalId() const;
  sro::Position position() const;
  uint16_t angle() const;
private:
  uint32_t globalId_;
  sro::Position position_;
  uint16_t angle_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POSITION_HPP