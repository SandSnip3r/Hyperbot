#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_REMOVE_OWNERSHIP_HPP_
#define PACKET_PARSING_SERVER_AGENT_ENTITY_REMOVE_OWNERSHIP_HPP_

#include "packet/parsing/parsedPacket.hpp"

#include <silkroad_lib/scalar_types.hpp>

namespace packet::parsing {

class ServerAgentEntityRemoveOwnership : public ParsedPacket {
public:
  ServerAgentEntityRemoveOwnership(const PacketContainer &packet);
  sro::scalar_types::EntityGlobalId globalId() const;
private:
  sro::scalar_types::EntityGlobalId globalId_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_REMOVE_OWNERSHIP_HPP_