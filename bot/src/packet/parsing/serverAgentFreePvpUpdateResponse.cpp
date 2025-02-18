#include "serverAgentFreePvpUpdateResponse.hpp"

namespace packet::parsing {

ServerAgentFreePvpUpdateResponse::ServerAgentFreePvpUpdateResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  if (result_ == 1) {
    stream.Read(globalId_);
    stream.Read(mode_);
  } else if (result_ == 2) {
    stream.Read(errorCode_);
  }
}

} // namespace packet::parsing