#ifndef PACKET_BUILDING_CLIENT_GATEWAY_SHARD_LIST_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_GATEWAY_SHARD_LIST_REQUEST_HPP_

#include "packet/opcode.hpp"

#include "../../shared/silkroad_security.h"

namespace packet::building {

class ClientGatewayShardListRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientGatewayShardListRequest;
  static const bool kEncrypted_ = true;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet();
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_GATEWAY_SHARD_LIST_REQUEST_HPP_