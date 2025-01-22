#ifndef PACKET_BUILDING_CLIENT_AGENT_SKILL_MASTERY_LEARN_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_SKILL_MASTERY_LEARN_REQUEST_HPP_

#include "packet/opcode.hpp"

#include "shared/silkroad_security.h"

#include <silkroad_lib/scalar_types.hpp>

namespace packet::building {

class ClientAgentSkillMasteryLearnRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentSkillMasteryLearnRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer incrementLevel(sro::scalar_types::ReferenceMasteryId masteryId);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_SKILL_MASTERY_LEARN_REQUEST_HPP_