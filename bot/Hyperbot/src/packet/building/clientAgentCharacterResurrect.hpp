#ifndef PACKET_BUILDING_CLIENT_AGENT_CHARACTER_RESURRECT_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_CHARACTER_RESURRECT_HPP_

#include "packet/opcode.hpp"
#include "packet/enums/packetEnums.hpp"

#include "../../shared/silkroad_security.h"

namespace packet::building {

class ClientAgentCharacterResurrect {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentCharacterResurrect;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer resurrect(packet::enums::ResurrectionOptionFlag option);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_CHARACTER_RESURRECT_HPP_