#include "commonParsing.hpp"
#include "serverAgentSkillEnd.hpp"

namespace packet::parsing {

ServerAgentSkillEnd::ServerAgentSkillEnd(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  if (result_ == 1) {
    stream.Read(castId_);
    stream.Read(targetGlobalId_);
    action_ = parseSkillAction(stream);
  } else if (result_ == 2) {
    stream.Read(errorCode_);
    stream.Read(castId_);
  }
}

uint8_t ServerAgentSkillEnd::result() const {
  return result_;
}

uint32_t ServerAgentSkillEnd::castId() const {
  return castId_;
}

sro::scalar_types::EntityGlobalId ServerAgentSkillEnd::targetGlobalId() const {
  return targetGlobalId_;
}

structures::SkillAction ServerAgentSkillEnd::action() const {
  return action_;
}

uint16_t ServerAgentSkillEnd::errorCode() const {
  return errorCode_;
}

} // namespace packet::parsing