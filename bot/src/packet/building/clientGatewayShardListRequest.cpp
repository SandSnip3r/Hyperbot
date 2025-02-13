#include "clientGatewayShardListRequest.hpp"

namespace packet::building {

PacketContainer ClientGatewayShardListRequest::packet() {
  StreamUtility stream;
  // Empty
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building