#include "clientAgentFreePvpUpdateRequest.hpp"

namespace packet::building {

PacketContainer ClientAgentFreePvpUpdateRequest::setMode(enums::FreePvpMode mode) {
  StreamUtility stream;
  stream.Write(mode);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building