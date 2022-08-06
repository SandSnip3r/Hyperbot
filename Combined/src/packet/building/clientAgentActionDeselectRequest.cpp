#include "clientAgentActionDeselectRequest.hpp"

namespace packet::building {

PacketContainer ClientAgentActionDeselectRequest::packet(uint32_t gId) {
  StreamUtility stream;
  stream.Write<uint32_t>(gId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building