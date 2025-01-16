#include "serverAgentAuthResponse.hpp"

namespace packet::parsing {

ServerAgentAuthResponse::ServerAgentAuthResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  if (result_ == 2) {
    stream.Read(errorCode_);
  }
}

} // namespace packet::parsing