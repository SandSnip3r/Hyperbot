#include "serverAgentEntityUpdateMoveSpeed.hpp"

namespace packet::parsing {

ServerAgentEntityUpdateMoveSpeed::ServerAgentEntityUpdateMoveSpeed(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<sro::scalar_types::EntityGlobalId>();
  walkSpeed_ = stream.Read<float>();
  runSpeed_ = stream.Read<float>();
}

sro::scalar_types::EntityGlobalId ServerAgentEntityUpdateMoveSpeed::globalId() const {
  return globalId_;
}

float ServerAgentEntityUpdateMoveSpeed::walkSpeed() const {
  return walkSpeed_;
}

float ServerAgentEntityUpdateMoveSpeed::runSpeed() const {
  return runSpeed_;
}

} // namespace packet::parsing