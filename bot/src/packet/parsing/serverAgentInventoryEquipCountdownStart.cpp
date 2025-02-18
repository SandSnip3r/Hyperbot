#include "serverAgentInventoryEquipCountdownStart.hpp"

namespace packet::parsing {

ServerAgentInventoryEquipCountdownStart::ServerAgentInventoryEquipCountdownStart(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(globalId_);
  // TODO: There is more, but we don't care yet.
}

} // namespace packet::parsing