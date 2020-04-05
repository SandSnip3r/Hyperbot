#include "../opcode.hpp"
#include "../../shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_AGENT_ACTION_SELECT_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_ACTION_SELECT_REQUEST_HPP

namespace packet::building {

class ClientAgentActionSelectRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentActionSelectRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(uint32_t gId);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_ACTION_SELECT_REQUEST_HPP