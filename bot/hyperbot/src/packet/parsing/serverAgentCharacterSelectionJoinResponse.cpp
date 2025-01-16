#include "serverAgentCharacterSelectionJoinResponse.hpp"

namespace packet::parsing {

ServerAgentCharacterSelectionJoinResponse::ServerAgentCharacterSelectionJoinResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  if (result_ == 2) {
    stream.Read(errorCode_);
  }
}

} // namespace packet::parsing