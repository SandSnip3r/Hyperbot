#include "serverAgentBuffAdd.hpp"

namespace packet::parsing {

ServerAgentBuffAdd::ServerAgentBuffAdd(const PacketContainer &packet, const pk2::SkillData &skillData) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(globalId_);
  stream.Read(skillRefId_);
  stream.Read(activeBuffToken_);

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

} // namespace packet::parsing