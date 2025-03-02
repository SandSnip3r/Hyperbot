#include "serverAgentEntityUpdateHwanLevel.hpp"

namespace packet::building {

PacketContainer ServerAgentEntityUpdateHwanLevel::packet(sro::scalar_types::EntityGlobalId globalId, uint8_t hwanLevel) {
  StreamUtility stream;
  stream.Write(globalId);
  stream.Write(hwanLevel);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building