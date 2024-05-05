#include "clientAgentActionDeselectRequest.hpp"

namespace packet::parsing {

ClientAgentActionDeselectRequest::ClientAgentActionDeselectRequest(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  gId_ = stream.Read<uint32_t>();
}

uint32_t ClientAgentActionDeselectRequest::gId() const {
  return gId_;
}

} // namespace packet::parsing