#include "clientAgentCharacterSelectionJoinRequest.hpp"

namespace packet::building {

PacketContainer ClientAgentCharacterSelectionJoinRequest::packet(const std::string &name) {
  StreamUtility stream;
  stream.Write<uint16_t>(static_cast<uint16_t>(name.size()));
  stream.Write_Ascii(name);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building