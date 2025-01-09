#include "../opcode.hpp"
#include "../../shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_AGENT_CHARACTER_SELECTION_JOIN_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_CHARACTER_SELECTION_JOIN_REQUEST_HPP

namespace packet::building {

class ClientAgentCharacterSelectionJoinRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentCharacterSelectionJoinRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(const std::string &name);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_CHARACTER_SELECTION_JOIN_REQUEST_HPP