#ifndef PACKET_BUILDING_CLIENT_AGENT_INVENTORY_OPERATION_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_AGENT_INVENTORY_OPERATION_REQUEST_HPP_

#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

#include <silkroad_lib/scalar_types.hpp>

#include <cstdint>

namespace packet::building {

class ClientAgentInventoryOperationRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentInventoryOperationRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(sro::scalar_types::StorageIndexType srcSlot, sro::scalar_types::StorageIndexType destSlot, uint16_t quantity);
  static PacketContainer packet(uint64_t goldDropAmount);
  static PacketContainer inventoryToStoragePacket(sro::scalar_types::StorageIndexType srcSlot, sro::scalar_types::StorageIndexType destSlot, uint32_t npcGId);
  static PacketContainer inventoryToAvatarPacket(sro::scalar_types::StorageIndexType srcSlot, sro::scalar_types::StorageIndexType destSlot);
  static PacketContainer avatarToInventoryPacket(sro::scalar_types::StorageIndexType srcSlot, sro::scalar_types::StorageIndexType destSlot);
  static PacketContainer withinInventoryPacket(sro::scalar_types::StorageIndexType srcSlot, sro::scalar_types::StorageIndexType destSlot, uint16_t quantity);
  static PacketContainer withinStoragePacket(sro::scalar_types::StorageIndexType srcSlot, sro::scalar_types::StorageIndexType destSlot, uint16_t quantity, uint32_t npcGId);
  static PacketContainer buyPacket(uint8_t tabIndex, uint8_t itemIndex, uint16_t quantity, uint32_t npcGId);
  static PacketContainer sellPacket(uint8_t itemIndex, uint16_t quantity, uint32_t npcGId);
  static PacketContainer dropItem(sro::scalar_types::StorageIndexType slot);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_INVENTORY_OPERATION_REQUEST_HPP_