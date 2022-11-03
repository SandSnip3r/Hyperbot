#include "serverAgentBuffAdd.hpp"

namespace packet::parsing {

ServerAgentBuffAdd::ServerAgentBuffAdd(const PacketContainer &packet, const pk2::SkillData &skillData) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<sro::scalar_types::EntityGlobalId>();
  skillRefId_ = stream.Read<sro::scalar_types::ReferenceObjectId>();
  activeBuffToken_ = stream.Read<uint32_t>();

  const auto &skill = skillData.getSkillById(skillRefId_);
  if (skill.isEfta()) {
    uint8_t creatorFlag = stream.Read<uint8_t>(); // 1=Creator, 2=Other
  }

  // getv
  // ROUGE_POISON_BUFF_UP
  // STEALTH_DURATION_UP
  // DOT_DURATION_UP
  // uint32_t continuousHoursIncreased = stream.Read<uint32_t>(); // In seconds
}

sro::scalar_types::EntityGlobalId ServerAgentBuffAdd::globalId() const {
  return globalId_;
}

sro::scalar_types::ReferenceObjectId ServerAgentBuffAdd::skillRefId() const {
  return skillRefId_;
}

uint32_t ServerAgentBuffAdd::activeBuffToken() const {
  return activeBuffToken_;
}

} // namespace packet::parsing