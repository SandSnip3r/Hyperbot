#ifndef PACKET_BUILDING_SERVER_AGENT_ENTITY_UPDATE_STATE_HPP_
#define PACKET_BUILDING_SERVER_AGENT_ENTITY_UPDATE_STATE_HPP_

#include "packet/opcode.hpp"

#include "shared/silkroad_security.h"

#include <silkroad_lib/entity.h>
#include <silkroad_lib/scalar_types.h>

namespace packet::building {

class ServerAgentEntityUpdateState {
private:
  static const Opcode kOpcode_ = Opcode::kServerAgentEntityUpdateState;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer updateLifeState(sro::scalar_types::EntityGlobalId globalId, sro::entity::LifeState state);
};

} // namespace packet::building

#endif // PACKET_BUILDING_SERVER_AGENT_ENTITY_UPDATE_STATE_HPP_