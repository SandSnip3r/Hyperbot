#include "commonParsing.hpp"
#include "serverAgentEntityUpdatePosition.hpp"

namespace packet::parsing {

ServerAgentEntityUpdatePosition::ServerAgentEntityUpdatePosition(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<uint32_t>();
  position_ = parsePosition(stream);
  angle_ = stream.Read<uint16_t>();
}

uint32_t ServerAgentEntityUpdatePosition::globalId() const {
  return globalId_;
}

sro::Position ServerAgentEntityUpdatePosition::position() const {
  return position_;
}

uint16_t ServerAgentEntityUpdatePosition::angle() const {
  return angle_;
}

} // namespace packet::parsing