#ifndef PACKET_BUILDING_SERVER_AGENT_ENTITY_UPDATE_HWAN_LEVEL_HPP_
#define PACKET_BUILDING_SERVER_AGENT_ENTITY_UPDATE_HWAN_LEVEL_HPP_

#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

#include <silkroad_lib/scalar_types.hpp>

namespace packet::building {

class ServerAgentEntityUpdateHwanLevel {
private:
  static const Opcode kOpcode_ = Opcode::kServerAgentEntityUpdateHwanLevel;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(sro::scalar_types::EntityGlobalId globalId, uint8_t hwanLevel);
};

} // namespace packet::building

#endif // PACKET_BUILDING_SERVER_AGENT_ENTITY_UPDATE_HWAN_LEVEL_HPP_