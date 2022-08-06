#include "clientAgentActionSelectRequest.hpp"

namespace packet::parsing {

ClientAgentActionSelectRequest::ClientAgentActionSelectRequest(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  gId_ = stream.Read<uint32_t>();
}

uint32_t ClientAgentActionSelectRequest::gId() const {
  return gId_;
}

} // namespace packet::parsing