#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POSITION_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POSITION_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"
#include "../structures/packetInnerStructures.hpp"

#include <cstdint>

namespace packet::parsing {
  
class ServerAgentEntityUpdatePosition : public ParsedPacket {
public:
  ServerAgentEntityUpdatePosition(const PacketContainer &packet);
  uint32_t globalId() const;
  structures::Position position() const;
  uint16_t angle() const;
private:
  uint32_t globalId_;
  structures::Position position_;
  uint16_t angle_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POSITION_HPP