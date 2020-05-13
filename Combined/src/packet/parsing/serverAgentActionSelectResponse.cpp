#include "serverAgentActionSelectResponse.hpp"

namespace packet::parsing {

ServerAgentActionSelectResponse::ServerAgentActionSelectResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = stream.Read<uint8_t>();
  if (result_ == 1) {
    gId_ = stream.Read<uint32_t>(); 
  } else {
    // Code 4 for items, or npc/player too far (but still in view)
    errorCode_ = stream.Read<uint16_t>();
  }
}

uint8_t ServerAgentActionSelectResponse::result() const {
  return result_;
}

uint32_t ServerAgentActionSelectResponse::gId() const {
  return gId_;
}

uint16_t ServerAgentActionSelectResponse::errorCode() const {
  return errorCode_;
}

} // namespace packet::parsing