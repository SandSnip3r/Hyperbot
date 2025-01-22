#ifndef PACKET_BUILDING_SERVER_AGENT_SKILL_END_HPP_
#define PACKET_BUILDING_SERVER_AGENT_SKILL_END_HPP_

#include "packet/opcode.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include "shared/silkroad_security.h"

#include <silkroad_lib/scalar_types.hpp>

namespace packet::building {

// This function was written for a hack and may not have full functionality
// TODO: Finish
class ServerAgentSkillEnd {
private:
  static const Opcode kOpcode_ = Opcode::kServerAgentSkillEnd;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  struct Input {
    uint32_t castId;
    sro::scalar_types::EntityGlobalId targetGlobalId;
    structures::SkillAction skillAction;
  };

  static PacketContainer packet(const Input &data);
};

} // namespace packet::building

#endif // PACKET_BUILDING_SERVER_AGENT_SKILL_END_HPP_