#include "serverAgentEntityUpdateExperience.hpp"

#include "../enums/packetEnums.hpp"

#include <type_traits>

namespace packet::parsing {

ServerAgentEntityUpdateExperience::ServerAgentEntityUpdateExperience(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint32_t sourceGlobalId = stream.Read<uint32_t>();
  stream.Read(gainedExperiencePoints_);
  stream.Read(gainedSpExperiencePoints_);
  uint8_t academyBuffUpdateBitmask = stream.Read<uint8_t>();
  if (academyBuffUpdateBitmask & static_cast<std::underlying_type_t<enums::AcademyBuffUpdateFlag>>(enums::AcademyBuffUpdateFlag::kCumulatedSize)) {
    uint32_t cumulatedSize = stream.Read<uint32_t>();
  }
  if (academyBuffUpdateBitmask & static_cast<std::underlying_type_t<enums::AcademyBuffUpdateFlag>>(enums::AcademyBuffUpdateFlag::kAccumulatedSize)) {
    uint32_t sourceCharacterId = stream.Read<uint32_t>();
    uint32_t accumulatedSize = stream.Read<uint32_t>();
  }
  // TODO: We need to somehow reference our character's level
  // If the gained experience points are > the max experience, we need to parse in a uint16_t for the stat points
}

int64_t ServerAgentEntityUpdateExperience::gainedExperiencePoints() const {
  return gainedExperiencePoints_;
}

uint64_t ServerAgentEntityUpdateExperience::gainedSpExperiencePoints() const {
  return gainedSpExperiencePoints_;
}

} // namespace packet::parsing