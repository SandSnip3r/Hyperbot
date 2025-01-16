#ifndef PACKET_BUILDING_CLIENT_AGENT_COS_COMMAND_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_COS_COMMAND_REQUEST_HPP_

#include "packet/opcode.hpp"

#include "../../shared/silkroad_security.h"

#include <silkroad_lib/scalar_types.h>

namespace packet::building {

class ClientAgentCosCommandRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentCosCommandRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer pickup(sro::scalar_types::EntityGlobalId cosGlobalId, sro::scalar_types::EntityGlobalId targetGlobalId);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_COS_COMMAND_REQUEST_HPP_