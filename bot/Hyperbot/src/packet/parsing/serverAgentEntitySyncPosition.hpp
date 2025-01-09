#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_SYNC_POSITION_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_SYNC_POSITION_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"
#include "../structures/packetInnerStructures.hpp"

#include <silkroad_lib/position.h>

#include <cstdint>

namespace packet::parsing {

class ServerAgentEntitySyncPosition : public ParsedPacket {
public:
  ServerAgentEntitySyncPosition(const PacketContainer &packet);
  sro::Position position() const;
  uint16_t angle() const;
  uint32_t globalId() const;
private:
  sro::Position position_;
  uint16_t angle_;
  uint32_t globalId_;

};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_SYNC_POSITION_HPP