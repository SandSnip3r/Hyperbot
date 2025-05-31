#ifndef PACKET_BUILDING_CLIENT_AGENT_GAME_RESET_COMPLETE_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_GAME_RESET_COMPLETE_HPP_

#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

namespace packet::building {

class ClientAgentGameResetComplete {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentGameResetComplete;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet();
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_GAME_RESET_COMPLETE_HPP_