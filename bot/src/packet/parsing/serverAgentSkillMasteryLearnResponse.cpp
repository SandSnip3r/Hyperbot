#include "serverAgentSkillMasteryLearnResponse.hpp"

#include <stdexcept>

namespace packet::parsing {

ServerAgentSkillMasteryLearnResponse::ServerAgentSkillMasteryLearnResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  if (result_ == 1) {
    stream.Read(masteryId_);
    stream.Read(masteryLevel_);
  } else if (result_ == 2) {
    stream.Read(errorCode_);
  } else {
    throw std::runtime_error("Unexpected result value");
  }
}

bool ServerAgentSkillMasteryLearnResponse::success() const {
  return result_ == 1;
}

uint32_t ServerAgentSkillMasteryLearnResponse::masteryId() const {
  return masteryId_;
}

uint8_t ServerAgentSkillMasteryLearnResponse::masteryLevel() const {
  return masteryLevel_;
}

uint16_t ServerAgentSkillMasteryLearnResponse::errorCode() const {
  return errorCode_;
}

} // namespace packet::parsing