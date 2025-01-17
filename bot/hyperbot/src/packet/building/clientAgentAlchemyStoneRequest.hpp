#ifndef PACKET_BUILDING_CLIENT_AGENT_ALCHEMY_STONE_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_ALCHEMY_STONE_REQUEST_HPP_

#include "packet/opcode.hpp"

#include "shared/silkroad_security.h"

#include <silkroad_lib/scalar_types.h>

namespace packet::building {

// TODO: This file is copied and pasted from ClientAgentAlchemyElixirRequest. They're the same, apart from the opcode.

class ClientAgentAlchemyStoneRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentAlchemyStoneRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer fuseStone(sro::scalar_types::StorageIndexType targetIndex, sro::scalar_types::StorageIndexType stoneIndex);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_ALCHEMY_STONE_REQUEST_HPP_