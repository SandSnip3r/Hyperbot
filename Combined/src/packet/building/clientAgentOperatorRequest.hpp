#include "packet/opcode.hpp"

#include "../../shared/silkroad_security.h"

#include <silkroad_lib/position.h>

#ifndef PACKET_BUILDING_CLIENT_AGENT_OPERATOR_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_OPERATOR_REQUEST_HPP_

namespace packet::building {

class ClientAgentOperatorRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentOperatorRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer toggleInvisible();
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_OPERATOR_REQUEST_HPP_