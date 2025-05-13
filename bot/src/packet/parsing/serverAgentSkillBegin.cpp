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
  if (result_ == 2) {
    throw std::runtime_error("ServerAgentSkillBeing: Error occurred when casting skill; refSkillId is not set");
  }
  return refSkillId_;
}

sro::scalar_types::EntityGlobalId ServerAgentSkillBegin::casterGlobalId() const {
  if (result_ == 2) {
    throw std::runtime_error("ServerAgentSkillBeing: Error occurred when casting skill; casterGlobalId is not set");
  }
  return casterGlobalId_;
}

uint32_t ServerAgentSkillBegin::castId() const {
  if (result_ == 2) {
    throw std::runtime_error("ServerAgentSkillBeing: Error occurred when casting skill; castId is not set");
  }
  return castId_;
}

sro::scalar_types::EntityGlobalId ServerAgentSkillBegin::targetGlobalId() const {
  if (result_ == 2) {
    throw std::runtime_error("ServerAgentSkillBeing: Error occurred when casting skill; targetGlobalId is not set");
  }
  return targetGlobalId_;
}

structures::SkillAction ServerAgentSkillBegin::action() const {
  if (result_ == 2) {
    throw std::runtime_error("ServerAgentSkillBeing: Error occurred when casting skill; action is not set");
  }
  return action_;
}

} // namespace packet::parsing