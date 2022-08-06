#include "serverAgentActionDeselectResponse.hpp"

namespace packet::parsing {

ServerAgentActionDeselectResponse::ServerAgentActionDeselectResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = stream.Read<uint8_t>();
  if (result_ == 2) {
    errorCode_ = stream.Read<uint16_t>();
  }
}

uint8_t ServerAgentActionDeselectResponse::result() const {
  return result_;
}

uint16_t ServerAgentActionDeselectResponse::errorCode() const {
  return errorCode_;
}

} // namespace packet::parsing