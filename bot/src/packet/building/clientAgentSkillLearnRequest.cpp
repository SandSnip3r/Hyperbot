#include "clientAgentSkillLearnRequest.hpp"

namespace packet::building {

PacketContainer ClientAgentSkillLearnRequest::learnSkill(sro::scalar_types::ReferenceSkillId skillId) {
  StreamUtility stream;
  stream.Write(skillId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building