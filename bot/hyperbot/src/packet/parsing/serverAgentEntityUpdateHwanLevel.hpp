#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_HWAN_LEVEL_HPP_
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_HWAN_LEVEL_HPP_

#include "packet/parsing/parsedPacket.hpp"

#include <silkroad_lib/scalar_types.h>

namespace packet::parsing {

class ServerAgentEntityUpdateHwanLevel : public ParsedPacket {
public:
  ServerAgentEntityUpdateHwanLevel(const PacketContainer &packet);
  sro::scalar_types::EntityGlobalId globalId() const;
  uint8_t hwanLevel() const;
private:
  sro::scalar_types::EntityGlobalId globalId_;
  uint8_t hwanLevel_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_HWAN_LEVEL_HPP_