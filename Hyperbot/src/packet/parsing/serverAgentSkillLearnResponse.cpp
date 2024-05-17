#include "serverAgentSkillLearnResponse.hpp"

namespace packet::parsing {

ServerAgentSkillLearnResponse::ServerAgentSkillLearnResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  if (result_ == 1) {
    stream.Read(skillId_);
  } else if (result_ == 2) {
    stream.Read(errorCode_);
  } else {
    throw std::runtime_error("Unexpected result value");
  }
}

bool ServerAgentSkillLearnResponse::success() const {
  return result_ == 1;
}

uint32_t ServerAgentSkillLearnResponse::skillId() const {
  return skillId_;
}

uint16_t ServerAgentSkillLearnResponse::errorCode() const {
  return errorCode_;
}

} // namespace packet::parsing