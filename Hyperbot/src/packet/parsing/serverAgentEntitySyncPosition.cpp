#include "commonParsing.hpp"
#include "serverAgentEntitySyncPosition.hpp"

namespace packet::parsing {

ServerAgentEntitySyncPosition::ServerAgentEntitySyncPosition(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  position_ = parsePosition(stream);
  angle_ = stream.Read<uint16_t>();
  globalId_ = stream.Read<uint32_t>();
}

uint32_t ServerAgentEntitySyncPosition::globalId() const {
  return globalId_;
}

sro::Position ServerAgentEntitySyncPosition::position() const {
  return position_;
}

uint16_t ServerAgentEntitySyncPosition::angle() const {
  return angle_;
}

} // namespace packet::parsing