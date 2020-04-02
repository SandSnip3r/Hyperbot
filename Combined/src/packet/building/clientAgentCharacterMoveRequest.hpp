#include "../opcode.hpp"
#include "../../shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_AGENT_CHARACTER_MOVE_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_CHARACTER_MOVE_REQUEST_HPP

namespace packet::building {

class ClientAgentCharacterMoveRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentCharacterMoveRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(uint16_t angle);
  static PacketContainer packet(uint16_t regionId, uint32_t xOffset, uint32_t yOffset, uint32_t zOffset);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_CHARACTER_MOVE_REQUEST_HPP