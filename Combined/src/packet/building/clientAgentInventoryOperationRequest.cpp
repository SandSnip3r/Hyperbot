#include "clientAgentInventoryOperationRequest.hpp"
#include "../enums/packetEnums.hpp"

namespace packet::building {

PacketContainer ClientAgentInventoryOperationRequest::packet(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(packet::enums::ItemMovementType::kWithinInventory));
  stream.Write<uint8_t>(srcSlot);
  stream.Write<uint8_t>(destSlot);
  stream.Write<uint16_t>(quantity);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentInventoryOperationRequest::packet(uint64_t goldDropAmount) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(packet::enums::ItemMovementType::kGoldDrop));
  stream.Write<>(goldDropAmount);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building