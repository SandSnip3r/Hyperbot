#include "serverAgentEntityUpdateHwanLevel.hpp"

namespace packet::parsing {

ServerAgentEntityUpdateHwanLevel::ServerAgentEntityUpdateHwanLevel(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(globalId_);
  stream.Read(hwanLevel_);
}

sro::scalar_types::EntityGlobalId ServerAgentEntityUpdateHwanLevel::globalId() const {
  return globalId_;
}

uint8_t ServerAgentEntityUpdateHwanLevel::hwanLevel() const {
  return hwanLevel_;
}

} // namespace packet::parsing