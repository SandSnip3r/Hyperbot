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
  static PacketContainer moveTowardAngle(uint16_t angle);
  static PacketContainer moveToPosition(uint16_t regionId, uint32_t xOffset, uint32_t yOffset, uint32_t zOffset);

  // The packet requires integers, we dont want to implicitly cast floats for this packet
  template<typename T>
  static PacketContainer moveToPosition(uint16_t regionId, T xOffset, T yOffset, T zOffset) = delete;
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_CHARACTER_MOVE_REQUEST_HPP