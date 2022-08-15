#include "serverAgentInventoryUpdateDurability.hpp"

namespace packet::parsing {

ServerAgentInventoryUpdateDurability::ServerAgentInventoryUpdateDurability(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  slotIndex_ = stream.Read<uint8_t>();
  durability_ = stream.Read<uint32_t>();
}

uint8_t ServerAgentInventoryUpdateDurability::slotIndex() const {
  return slotIndex_;
}

uint32_t ServerAgentInventoryUpdateDurability::durability() const {
  return durability_;
}

} // namespace packet::parsing