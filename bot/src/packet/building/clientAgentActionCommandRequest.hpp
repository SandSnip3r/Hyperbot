#ifndef PACKET_BUILDING_CLIENT_AGENT_ACTION_COMMAND_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_ACTION_COMMAND_REQUEST_HPP_

#include "packet/opcode.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "shared/silkroad_security.h"

#include <silkroad_lib/scalar_types.hpp>

namespace packet::building {

class ClientAgentActionCommandRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentActionCommandRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer cancel();
  static PacketContainer attack(sro::scalar_types::EntityGlobalId targetGId);
  static PacketContainer pickup(sro::scalar_types::EntityGlobalId targetGId);
  static PacketContainer trace(sro::scalar_types::EntityGlobalId targetGId);
  static PacketContainer cast(sro::scalar_types::ReferenceObjectId refSkillId);
  static PacketContainer cast(sro::scalar_types::ReferenceObjectId refSkillId, sro::scalar_types::EntityGlobalId targetGId);
  static PacketContainer dispel(sro::scalar_types::ReferenceObjectId refSkillId, sro::scalar_types::EntityGlobalId targetGId);
  static PacketContainer command(const structures::ActionCommand& actionCommand);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_ACTION_COMMAND_REQUEST_HPP_