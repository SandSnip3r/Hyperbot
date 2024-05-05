#include "commonBuilding.hpp"
#include "serverAgentSkillEnd.hpp"

namespace packet::building {

PacketContainer ServerAgentSkillEnd::packet(const ServerAgentSkillEnd::Input &data) {
  StreamUtility stream;
  stream.Write<uint8_t>(0x01); // result

  stream.Write<>(data.castId);
  stream.Write<>(data.targetGlobalId);

  writeSkillAction(stream, data.skillAction);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building