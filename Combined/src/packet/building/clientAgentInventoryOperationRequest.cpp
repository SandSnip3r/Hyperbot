#include "clientAgentInventoryOperationRequest.hpp"

#include "packet/enums/packetEnums.hpp"

namespace packet::building {

PacketContainer ClientAgentInventoryOperationRequest::packet(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(packet::enums::ItemMovementType::kUpdateSlotsInventory));
  stream.Write<uint8_t>(srcSlot);
  stream.Write<uint8_t>(destSlot);
  stream.Write<uint16_t>(quantity);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentInventoryOperationRequest::packet(uint64_t goldDropAmount) {
  StreamUtility stream;
  stream.Write<uint8_t>(static_cast<uint8_t>(packet::enums::ItemMovementType::kDropGold));
  stream.Write<>(goldDropAmount);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentInventoryOperationRequest::inventoryToStoragePacket(uint8_t srcSlot, uint8_t destSlot, uint32_t npcGId) {
  StreamUtility stream;
  stream.Write<>(static_cast<uint8_t>(packet::enums::ItemMovementType::kChestDepositItem));
  stream.Write<>(srcSlot);
  stream.Write<>(destSlot);
  stream.Write<>(npcGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentInventoryOperationRequest::withinInventoryPacket(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity) {
  StreamUtility stream;
  stream.Write<>(static_cast<uint8_t>(packet::enums::ItemMovementType::kUpdateSlotsInventory));
  stream.Write<>(srcSlot);
  stream.Write<>(destSlot);
  stream.Write<>(quantity);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentInventoryOperationRequest::withinStoragePacket(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity, uint32_t npcGId) {
  StreamUtility stream;
  stream.Write<>(static_cast<uint8_t>(packet::enums::ItemMovementType::kUpdateSlotsChest));
  stream.Write<>(srcSlot);
  stream.Write<>(destSlot);
  stream.Write<>(quantity);
  stream.Write<>(npcGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

PacketContainer ClientAgentInventoryOperationRequest::buyPacket(uint8_t tabIndex, uint8_t itemIndex, uint16_t quantity, uint32_t npcGId) {
  StreamUtility stream;
  stream.Write<>(static_cast<uint8_t>(packet::enums::ItemMovementType::kBuyItem));
  stream.Write<>(tabIndex);
  stream.Write<>(itemIndex);
  stream.Write<>(quantity);
  stream.Write<>(npcGId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building