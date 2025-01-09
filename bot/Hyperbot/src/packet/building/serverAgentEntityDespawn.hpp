#ifndef PACKET_BUILDING_SERVER_AGENT_ENTITY_DESPAWN_HPP_
#define PACKET_BUILDING_SERVER_AGENT_ENTITY_DESPAWN_HPP_

#include "packet/opcode.hpp"

#include "../../shared/silkroad_security.h"

#include <silkroad_lib/scalar_types.h>

namespace packet::building {

class ServerAgentEntityDespawn {
private:
  static const Opcode kOpcode_ = Opcode::kServerAgentEntityDespawn;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(sro::scalar_types::EntityGlobalId globalId);
};

} // namespace packet::building

#endif // PACKET_BUILDING_SERVER_AGENT_ENTITY_DESPAWN_HPP_