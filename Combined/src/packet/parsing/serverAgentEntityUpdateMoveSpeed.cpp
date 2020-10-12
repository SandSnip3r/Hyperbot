#include "serverAgentEntityUpdateMoveSpeed.hpp"

namespace packet::parsing {

ServerAgentEntityUpdateMoveSpeed::ServerAgentEntityUpdateMoveSpeed(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<uint32_t>();

  uint32_t walkSpeedBytes = stream.Read<uint32_t>();
  walkSpeed_ = *reinterpret_cast<float*>(&walkSpeedBytes);

  uint32_t runSpeedBytes = stream.Read<uint32_t>();
  runSpeed_ = *reinterpret_cast<float*>(&runSpeedBytes);
}

uint32_t ServerAgentEntityUpdateMoveSpeed::globalId() const {
  return globalId_;
}

float ServerAgentEntityUpdateMoveSpeed::walkSpeed() const {
  return walkSpeed_;
}

float ServerAgentEntityUpdateMoveSpeed::runSpeed() const {
  return runSpeed_;
}

} // namespace packet::parsing