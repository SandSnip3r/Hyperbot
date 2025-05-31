#include "clientAgentCharacterSelectionJoinRequest.hpp"

namespace packet::building {

PacketContainer ClientAgentCharacterSelectionJoinRequest::packet(const std::string &name) {
  StreamUtility stream;
  stream.Write(name);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building