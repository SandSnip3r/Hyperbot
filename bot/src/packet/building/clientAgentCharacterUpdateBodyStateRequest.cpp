#include "clientAgentCharacterUpdateBodyStateRequest.hpp"

namespace packet::building {

PacketContainer ClientAgentCharacterUpdateBodyStateRequest::packet(enums::BodyState bodyState) {
  StreamUtility stream;
  stream.Write<>(bodyState);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building