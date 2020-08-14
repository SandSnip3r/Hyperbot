#include "../opcode.hpp"
#include "../../shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_AGENT_ACTION_COMMAND_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_ACTION_COMMAND_REQUEST_HPP

namespace packet::building {

class ClientAgentActionCommandRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentActionCommandRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer cancel();
  static PacketContainer attack(uint32_t targetGId);
  static PacketContainer pickup(uint32_t targetGId);
  static PacketContainer trace(uint32_t targetGId);
  static PacketContainer cast(uint32_t refSkillId);
  static PacketContainer cast(uint32_t refSkillId, uint32_t targetGId);
  static PacketContainer dispel(uint32_t refSkillId, uint32_t targetGId);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_ACTION_COMMAND_REQUEST_HPP