#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_SYNC_POSITION_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_SYNC_POSITION_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"
#include "../structures/packetInnerStructures.hpp"

#include <cstdint>

namespace packet::parsing {

// kServerAgentEntitySyncPosition = 0x3028

class ServerAgentEntitySyncPosition : public ParsedPacket {
public:
  ServerAgentEntitySyncPosition(const PacketContainer &packet);
  structures::Position position() const;
  uint16_t angle() const;
  uint32_t globalId() const;
private:
  structures::Position position_;
  uint16_t angle_;
  uint32_t globalId_;

};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_SYNC_POSITION_HPP