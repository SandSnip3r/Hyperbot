#include "clientAgentSkillMasteryLearnRequest.hpp"

namespace packet::building {

PacketContainer ClientAgentSkillMasteryLearnRequest::incrementLevel(sro::scalar_types::ReferenceMasteryId masteryId) {
  StreamUtility stream;
  stream.Write(masteryId);
  constexpr uint8_t kLevelsToAdd{1};
  stream.Write(kLevelsToAdd); // Supposedly, this value is the number of levels to increase the mastery by, however, it seems that for any value, including 0, the mastery always increases by 1.
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building