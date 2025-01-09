#include "commonBuilding.hpp"
#include "serverAgentSkillBegin.hpp"

namespace packet::building {

PacketContainer ServerAgentSkillBegin::packet(const ServerAgentSkillBegin::Input &data) {
  StreamUtility stream;
  stream.Write<uint8_t>(0x01); // result
  stream.Write<uint16_t>(0x3002); // errorCode
  
  stream.Write<>(data.skillRefId);
  stream.Write<>(data.casterGlobalId);
  stream.Write<>(data.castId);
  stream.Write<>(data.targetGlobalId);

  writeSkillAction(stream, data.skillAction);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building