#include "serverAgentActionSelectResponse.hpp"

namespace packet::parsing {

ServerAgentActionSelectResponse::ServerAgentActionSelectResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  std::cout << "Parsing ServerAgentActionSelectResponse\n";
  StreamUtility stream = packet.data;
  result_ = stream.Read<uint8_t>();
  std::cout << "result: " << (int)result_ << '\n';
  if (result_ == 1) {
    gId_ = stream.Read<uint32_t>(); 
    std::cout << "gId: " << gId_ << '\n';
  } else {
    // Code 4 for items, or npc/player too far (but still in view)
    errorCode_ = stream.Read<uint16_t>();
    std::cout << "errorCode: " << errorCode_ << '\n';
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