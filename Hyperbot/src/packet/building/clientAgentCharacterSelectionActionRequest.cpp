#include "clientAgentCharacterSelectionActionRequest.hpp"

namespace packet::building {

PacketContainer ClientAgentCharacterSelectionActionRequest::packet(packet::enums::CharacterSelectionAction action) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(action));
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building