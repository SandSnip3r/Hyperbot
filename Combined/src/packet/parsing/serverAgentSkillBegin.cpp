#include "commonParsing.hpp"
#include "serverAgentSkillBegin.hpp"

#include "logging.hpp"

namespace packet::parsing {

ServerAgentSkillBegin::ServerAgentSkillBegin(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = stream.Read<uint8_t>();
  errorCode_ = stream.Read<uint16_t>();
  if (result_ == 2) {
    return;
  }

  refSkillId_ = stream.Read<uint32_t>();
  casterGlobalId_ = stream.Read<uint32_t>();
  castId_ = stream.Read<uint32_t>();
  targetGlobalId_ = stream.Read<uint32_t>();
  action_ = parseSkillAction(stream);
}

uint8_t ServerAgentSkillBegin::result() const {
  return result_;
}

uint16_t ServerAgentSkillBegin::errorCode() const {
  return errorCode_;
}

uint32_t ServerAgentSkillBegin::refSkillId() const {
  return refSkillId_;
}

uint32_t ServerAgentSkillBegin::casterGlobalId() const {
  return casterGlobalId_;
}

uint32_t ServerAgentSkillBegin::castId() const {
  return castId_;
}

uint32_t ServerAgentSkillBegin::targetGlobalId() const {
  return targetGlobalId_;
}

structures::SkillAction ServerAgentSkillBegin::action() const {
  return action_;
}

} // namespace packet::parsing