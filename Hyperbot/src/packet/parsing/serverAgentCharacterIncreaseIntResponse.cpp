#include "serverAgentCharacterIncreaseIntResponse.hpp"

namespace packet::parsing {

ServerAgentCharacterIncreaseIntResponse::ServerAgentCharacterIncreaseIntResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  if (result_ == 2) {
    // 29702 = (Not displayed) Not enough stat points.
    stream.Read(errorCode_);
  }
}

} // namespace packet::parsing