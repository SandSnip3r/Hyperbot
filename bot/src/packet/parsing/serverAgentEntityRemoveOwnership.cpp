#include "serverAgentEntityRemoveOwnership.hpp"

namespace packet::parsing {

ServerAgentEntityRemoveOwnership::ServerAgentEntityRemoveOwnership(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<sro::scalar_types::EntityGlobalId>();
}

sro::scalar_types::EntityGlobalId ServerAgentEntityRemoveOwnership::globalId() const {
  return globalId_;
}

} // namespace packet::parsing