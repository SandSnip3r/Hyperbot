#include "commonParsing.hpp"
#include "serverAgentSkillEnd.hpp"

namespace packet::parsing {

ServerAgentSkillEnd::ServerAgentSkillEnd(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = stream.Read<uint8_t>();
  if (result_ == 1) {
    castId_ = stream.Read<uint32_t>();
    targetGId_ = stream.Read<uint32_t>();
    action_ = parseSkillAction(stream);
  } else if (result_ == 2) {
    errorCode_ = stream.Read<uint16_t>();
    castId_ = stream.Read<uint32_t>();
  }
}

uint8_t ServerAgentSkillEnd::result() const {
  return result_;
}

uint32_t ServerAgentSkillEnd::castId() const {
  return castId_;
}

uint32_t ServerAgentSkillEnd::targetGId() const {
  return targetGId_;
}

structures::SkillAction ServerAgentSkillEnd::action() const {
  return action_;
}

uint16_t ServerAgentSkillEnd::errorCode() const {
  return errorCode_;
}

} // namespace packet::parsing