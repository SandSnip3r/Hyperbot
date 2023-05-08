#include "packet/opcode.hpp"

#include "../../shared/silkroad_security.h"

#include <silkroad_lib/position.h>

#ifndef PACKET_BUILDING_CLIENT_AGENT_CHARACTER_MOVE_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_CHARACTER_MOVE_REQUEST_HPP

namespace packet::building {

struct NetworkReadyPosition {
public:
  NetworkReadyPosition(const sro::Position &pos);
  sro::Position asSroPosition() const;
  static sro::Position truncateForNetwork(const sro::Position &pos);
private:
  sro::Position convertedPosition_;
};

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