#include "commonParsing.hpp"
#include "serverAgentSkillBegin.hpp"

namespace packet::parsing {

ServerAgentSkillBegin::ServerAgentSkillBegin(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  stream.Read(errorCode_);
  if (result_ == 2) {
    return;
  }
  stream.Read(refSkillId_);
  stream.Read(casterGlobalId_);
  stream.Read(castId_);
  stream.Read(targetGlobalId_);
  action_ = parseSkillAction(stream);
}

uint8_t ServerAgentSkillBegin::result() const {
  return result_;
}

uint16_t ServerAgentSkillBegin::errorCode() const {
  return errorCode_;
}

sro::scalar_types::ReferenceObjectId ServerAgentSkillBegin::refSkillId() const {
  return refSkillId_;
}

sro::scalar_types::EntityGlobalId ServerAgentSkillBegin::casterGlobalId() const {
  return casterGlobalId_;
}

uint32_t ServerAgentSkillBegin::castId() const {
  return castId_;
}

sro::scalar_types::EntityGlobalId ServerAgentSkillBegin::targetGlobalId() const {
  return targetGlobalId_;
}

structures::SkillAction ServerAgentSkillBegin::action() const {
  return action_;
}

} // namespace packet::parsing