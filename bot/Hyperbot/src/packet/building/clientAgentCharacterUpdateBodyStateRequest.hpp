#ifndef PACKET_BUILDING_CLIENT_AGENT_CHARACTER_UPDATE_BODY_STATE_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_CHARACTER_UPDATE_BODY_STATE_REQUEST_HPP_

#include "packet/enums/packetEnums.hpp"
#include "packet/opcode.hpp"

#include "../../shared/silkroad_security.h"

namespace packet::building {

class ClientAgentCharacterUpdateBodyStateRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentCharacterUpdateBodyStateRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(enums::BodyState bodyState);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_CHARACTER_UPDATE_BODY_STATE_REQUEST_HPP_