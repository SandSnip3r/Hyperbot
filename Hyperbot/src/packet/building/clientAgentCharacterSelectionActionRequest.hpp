#include "../enums/packetEnums.hpp"
#include "../opcode.hpp"
#include "../../shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_AGENT_CHARACTER_SELECTION_ACTION_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_CHARACTER_SELECTION_ACTION_REQUEST_HPP

namespace packet::building {

class ClientAgentCharacterSelectionActionRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentCharacterSelectionActionRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(packet::enums::CharacterSelectionAction action);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_CHARACTER_SELECTION_ACTION_REQUEST_HPP