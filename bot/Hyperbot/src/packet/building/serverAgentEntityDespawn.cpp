#include "serverAgentEntityDespawn.hpp"

namespace packet::building {

PacketContainer ServerAgentEntityDespawn::packet(sro::scalar_types::EntityGlobalId globalId) {
  StreamUtility stream;
  stream.Write<>(globalId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building