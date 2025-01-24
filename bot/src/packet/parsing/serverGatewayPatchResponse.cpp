#include "serverGatewayPatchResponse.hpp"

namespace packet::parsing {

ServerGatewayPatchResponse::ServerGatewayPatchResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  // TODO: There is more, but for now, we're not using any of it.
}

} // namespace packet::parsing