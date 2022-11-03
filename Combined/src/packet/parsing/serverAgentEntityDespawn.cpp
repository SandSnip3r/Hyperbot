#include "serverAgentEntityDespawn.hpp"

namespace packet::parsing {

ServerAgentEntityDespawn::ServerAgentEntityDespawn(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<sro::scalar_types::EntityGlobalId>();
}

uint32_t ServerAgentEntityDespawn::globalId() const {
  return globalId_;
}

} // namespace packet::parsing