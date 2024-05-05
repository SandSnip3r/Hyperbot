#include "clientAgentInventoryStorageOpenRequest.hpp"

namespace packet::building {

PacketContainer ClientAgentInventoryStorageOpenRequest::packet(uint32_t gId) {
  StreamUtility stream;
  stream.Write<uint32_t>(gId);
  stream.Write<uint8_t>(0); // Unknown byte
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building