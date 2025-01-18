#include "serverAgentEntityUpdateAngle.hpp"

namespace packet::parsing {

ServerAgentEntityUpdateAngle::ServerAgentEntityUpdateAngle(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<sro::scalar_types::EntityGlobalId>();
  angle_ = stream.Read<sro::Angle>();
}

sro::scalar_types::EntityGlobalId ServerAgentEntityUpdateAngle::globalId() const {
  return globalId_;
}

sro::Angle ServerAgentEntityUpdateAngle::angle() const {
  return angle_;
}

} // namespace packet::parsing