#include "commonParsing.hpp"
#include "parsedPacket.hpp"

#include <silkroad_lib/position_math.hpp>

#include <absl/log/log.h>

namespace packet::parsing {

//=========================================================================================================================================================

ParsedPacket::ParsedPacket(const PacketContainer &packet) : opcode_(static_cast<Opcode>(packet.opcode)) {}

Opcode ParsedPacket::opcode() const {
  return opcode_;
}

ParsedPacket::~ParsedPacket() {}

//=========================================================================================================================================================

ParsedClientItemMove::ParsedClientItemMove(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  movement_.type = static_cast<packet::enums::ItemMovementType>(stream.Read<uint8_t>());
  if (movement_.type == packet::enums::ItemMovementType::kUpdateSlotsInventory) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kUpdateSlotsChest ||
             movement_.type == packet::enums::ItemMovementType::kUpdateSlotsGuildChest) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
    uint32_t unk0 = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kChestDepositItem ||
             movement_.type == packet::enums::ItemMovementType::kChestWithdrawItem ||
             movement_.type == packet::enums::ItemMovementType::kGuildChestDepositItem ||
             movement_.type == packet::enums::ItemMovementType::kGuildChestWithdrawItem) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint32_t unk0 = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kBuyItem) {
    movement_.storeTabNumber = stream.Read<uint8_t>();
    movement_.storeSlotNumber = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
    movement_.globalId = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kSellItem) {
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
    uint32_t unk1 = stream.Read<uint32_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kDropGold ||
             movement_.type == packet::enums::ItemMovementType::kChestWithdrawGold ||
             movement_.type == packet::enums::ItemMovementType::kChestDepositGold ||
             movement_.type == packet::enums::ItemMovementType::kGuildChestDepositGold ||
             movement_.type == packet::enums::ItemMovementType::kGuildChestWithdrawGold) {
    uint64_t goldAmount = stream.Read<uint64_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kMoveItemCosToInventory ||
             movement_.type == packet::enums::ItemMovementType::kMoveItemInventoryToCos) {
    uint32_t unk4 = stream.Read<uint32_t>();
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kUpdateSlotsInventoryCos) {
    uint32_t unk4 = stream.Read<uint32_t>();
    uint8_t sourceSlot = stream.Read<uint8_t>();
    uint8_t destSlot = stream.Read<uint8_t>();
    uint16_t quantity = stream.Read<uint16_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kMoveItemAvatarToInventory) {
    uint8_t sourceAvatarInventorySlot = stream.Read<uint8_t>();
    uint8_t destInventorySlot = stream.Read<uint8_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kMoveItemInventoryToAvatar) {
    uint8_t sourceInventorySlot = stream.Read<uint8_t>();
    uint8_t destAvatarInventorySlot = stream.Read<uint8_t>();
  } else if (movement_.type == packet::enums::ItemMovementType::kDropItem) {
    uint8_t sourceInventorySlot = stream.Read<uint8_t>();
  } else {
    LOG(INFO) << "New item movement type! " << static_cast<int>(movement_.type);
    LOG(INFO) << "Dump: " << DumpToString(stream);
  }
}

structures::ItemMovement ParsedClientItemMove::movement() const {
  return movement_;
}

//=========================================================================================================================================================

} // namespace packet::parsing