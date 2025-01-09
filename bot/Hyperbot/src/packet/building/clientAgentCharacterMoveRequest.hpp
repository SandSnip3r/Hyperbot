#include "packet/building/commonBuilding.hpp"
#include "packet/opcode.hpp"

#include "../../shared/silkroad_security.h"

#include <silkroad_lib/position.h>

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
  static PacketContainer moveToPosition(const NetworkReadyPosition &newPosition);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_CHARACTER_MOVE_REQUEST_HPP