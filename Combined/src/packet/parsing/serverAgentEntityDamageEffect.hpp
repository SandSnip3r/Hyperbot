#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_DAMAGE_EFFECT_HPP_
#define PACKET_PARSING_SERVER_AGENT_ENTITY_DAMAGE_EFFECT_HPP_

#include "packet/parsing/parsedPacket.hpp"

#include <silkroad_lib/scalar_types.h>

namespace packet::parsing {

class ServerAgentEntityDamageEffect : public ParsedPacket {
public:
  ServerAgentEntityDamageEffect(const PacketContainer &packet);
  sro::scalar_types::EntityGlobalId globalId() const;
  uint32_t effectDamage() const;
private:
  sro::scalar_types::EntityGlobalId globalId_;
  uint32_t effectDamage_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_DAMAGE_EFFECT_HPP_