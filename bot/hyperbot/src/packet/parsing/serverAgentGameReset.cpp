#include "serverAgentGameReset.hpp"

namespace packet::parsing {

ServerAgentGameReset::ServerAgentGameReset(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
}

} // namespace packet::parsing