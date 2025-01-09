#include "serverGatewayLoginIbuvChallenge.hpp"

namespace packet::parsing {

ServerGatewayLoginIbuvChallenge::ServerGatewayLoginIbuvChallenge(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
}

} // namespace packet::parsing