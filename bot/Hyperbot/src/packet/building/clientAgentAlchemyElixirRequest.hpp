#ifndef PACKET_BUILDING_CLIENT_AGENT_ALCHEMY_ELIXIR_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_ALCHEMY_ELIXIR_REQUEST_HPP_

#include "packet/opcode.hpp"

#include "../../shared/silkroad_security.h"

#include <silkroad_lib/scalar_types.h>

namespace packet::building {

class ClientAgentAlchemyElixirRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentAlchemyElixirRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer fuseElixir(sro::scalar_types::StorageIndexType targetIndex, sro::scalar_types::StorageIndexType elixirIndex, std::vector<sro::scalar_types::StorageIndexType> enhancerIndices);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_ALCHEMY_ELIXIR_REQUEST_HPP_