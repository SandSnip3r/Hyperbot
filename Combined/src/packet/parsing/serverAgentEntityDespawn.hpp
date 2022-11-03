#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_DESPAWN_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_DESPAWN_HPP

#include "packet/parsing/parsedPacket.hpp"

#include <cstdint>

namespace packet::parsing {

class ServerAgentEntityDespawn : public ParsedPacket {
public:
  ServerAgentEntityDespawn(const PacketContainer &packet);
  uint32_t globalId() const;
private:
  uint32_t globalId_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_DESPAWN_HPP