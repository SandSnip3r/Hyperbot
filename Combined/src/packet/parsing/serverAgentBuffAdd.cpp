#include "serverAgentBuffAdd.hpp"

namespace packet::parsing {

ServerAgentBuffAdd::ServerAgentBuffAdd(const PacketContainer &packet, const pk2::SkillData &skillData) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<uint32_t>();
  skillRefId_ = stream.Read<uint32_t>();
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

uint32_t ServerAgentBuffAdd::globalId() const {
  return globalId_;
}

uint32_t ServerAgentBuffAdd::skillRefId() const {
  return skillRefId_;
}

uint32_t ServerAgentBuffAdd::activeBuffToken() const {
  return activeBuffToken_;
}

} // namespace packet::parsing