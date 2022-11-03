#include "packet/opcode.hpp"
#include "packet/enums/packetEnums.hpp"

#include "../../shared/silkroad_security.h"

#include <silkroad_lib/position.h>

#ifndef PACKET_BUILDING_CLIENT_AGENT_FREE_PVP_UPDATE_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_FREE_PVP_UPDATE_REQUEST_HPP_

namespace packet::building {

class ClientAgentFreePvpUpdateRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentFreePvpUpdateRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer setMode(enums::FreePvpMode mode);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_FREE_PVP_UPDATE_REQUEST_HPP_