
#include "packet/opcode.hpp"
#include "../../shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_AGENT_ACTION_DESELECT_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_ACTION_DESELECT_REQUEST_HPP_

namespace packet::building {

class ClientAgentActionDeselectRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentActionDeselectRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(uint32_t gId);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_ACTION_DESELECT_REQUEST_HPP_