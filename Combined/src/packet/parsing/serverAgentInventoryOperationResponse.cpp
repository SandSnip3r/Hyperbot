#include "serverAgentInventoryOperationResponse.hpp"

#include "packet/parsing/commonParsing.hpp"

#include <absl/log/log.h>

namespace packet::parsing {

const std::vector<structures::ItemMovement>& ServerAgentInventoryOperationResponse::itemMovements() const {
  return itemMovements_;
}

ServerAgentInventoryOperationResponse::ServerAgentInventoryOperationResponse(const PacketContainer &packet, const pk2::ItemData &itemData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint8_t result_ = stream.Read<uint8_t>();
  if (result_ == 1) {
    // Success
    structures::ItemMovement primaryItemMovement;
    primaryItemMovement.type = static_cast<packet::enums::ItemMovementType>(stream.Read<uint8_t>());
    if (primaryItemMovement.type == packet::enums::ItemMovementType::kUpdateSlotsInventory ||
        primaryItemMovement.type == packet::enums::ItemMovementType::kMoveItemAvatarToInventory ||
        primaryItemMovement.type == packet::enums::ItemMovementType::kMoveItemInventoryToAvatar) {
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>(); // Seems to be 0 when equiping or unequiping gear/avatars

      uint8_t secondaryMovementCount = stream.Read<uint8_t>();
      // While moving things around inside our inventory, there's a possibility that more items get moved too
      //  Like when we remove our dress, the accessory is forcefully removed
      for (int i=0; i<secondaryMovementCount; ++i) {
        structures::ItemMovement secondaryItemMovement;
        secondaryItemMovement.type = static_cast<packet::enums::ItemMovementType>(stream.Read<uint8_t>());
        // TODO: We assume that it will always be an inventory movement. However, it could
        //  technically be any kind of movement with the same data structure (ex. withinStorage)
        secondaryItemMovement.srcSlot = stream.Read<uint8_t>();
        secondaryItemMovement.destSlot = stream.Read<uint8_t>();
        secondaryItemMovement.quantity = stream.Read<uint16_t>(); // Seems to be 0 when equiping or unequiping gear/avatars
        itemMovements_.push_back(secondaryItemMovement);
      }
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kChestDepositItem ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kChestWithdrawItem ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kGuildChestDepositItem ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kGuildChestWithdrawItem) {
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kDropGold ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kChestWithdrawGold ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kChestDepositGold ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kGuildChestDepositGold ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kGuildChestWithdrawGold ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kSetExchangeGold) {
      primaryItemMovement.goldAmount = stream.Read<uint64_t>();
      if (primaryItemMovement.type == packet::enums::ItemMovementType::kSetExchangeGold) {
        LOG(INFO) << "This gold update is \"kSetExchangeGold\" with amount: " << primaryItemMovement.goldAmount;
      }
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kPickItem) {
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      if (primaryItemMovement.destSlot == structures::ItemMovement::kGoldSlot) {
        // Picked gold
        primaryItemMovement.goldPickAmount = stream.Read<uint32_t>();
      } else {
        // Picked an item
        primaryItemMovement.newItem = parseGenericItem(stream, itemData);
      }
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kDropItem) {
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kUpdateSlotsInventoryCos) {
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // COS global ID
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kAddItemByServer) {
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      uint8_t reason = stream.Read<uint8_t>();
      primaryItemMovement.newItem = parseGenericItem(stream, itemData);
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kRemoveItemByServer) {
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      uint8_t reason = stream.Read<uint8_t>();
      // Reason 2 happens when you use the last expendable in a stack
      // From Daxter: 02 = CashItemConsume, 03 = Skill Remove, 04 = Alchemy
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kMoveItemCosToInventory ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kMoveItemInventoryToCos) {
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // COS global ID
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kUpdateSlotsChest ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kUpdateSlotsGuildChest) {
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kBuyItem) {
      primaryItemMovement.storeTabNumber = stream.Read<uint8_t>();
      primaryItemMovement.storeSlotNumber = stream.Read<uint8_t>();
      uint8_t stackCount = stream.Read<uint8_t>();
      for (int i=0; i<stackCount; ++i) {
        // Can only happen multiple times if its an item that wont get stacked. Like equipment
        primaryItemMovement.destSlots.emplace_back(stream.Read<uint8_t>());
      }
      primaryItemMovement.quantity = stream.Read<uint16_t>();
      for (int i=0; i<stackCount; ++i) {
        primaryItemMovement.rentInfos.emplace_back(parseRentInfo(stream));
      }
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kSellItem) {
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>();
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // NPC global ID
      primaryItemMovement.buybackStackSize = stream.Read<uint8_t>(); // TODO: This might actually be target-slot, the NPC's buyback slot
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kBuyback) {
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.srcSlot = stream.Read<uint8_t>(); // Shop buyback slot, left is max(buybackStackSize-1), right is 0
      primaryItemMovement.quantity = stream.Read<uint16_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kPickItemByOther) {
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // COS?
      primaryItemMovement.destSlot = stream.Read<uint8_t>(); // Gold slot, always 0xFE
      if (primaryItemMovement.destSlot == 0xFE) {
        primaryItemMovement.goldPickAmount = stream.Read<uint32_t>();
      } else {
        LOG(INFO) << "Pick item by other. Slot: " << static_cast<int>(primaryItemMovement.destSlot);
        primaryItemMovement.newItem = parseGenericItem(stream, itemData);
      }
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kSetExchangeItem) {
      LOG(INFO) << "InventoryOperationResponse type kSetExchangeItem";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kCancelExchangeItem) {
      LOG(INFO) << "InventoryOperationResponse type kCancelExchangeItem";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kPickItemCos) {
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // COS
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      if (primaryItemMovement.destSlot == 0xFE) {
        // Not yet sure if this is happens
        throw std::runtime_error("COS picked item into gold slot");
      }
      primaryItemMovement.newItem = parseGenericItem(stream, itemData);
      uint16_t ownerNameLength = stream.Read<uint16_t>();
      std::string ownerName = stream.Read_Ascii(ownerNameLength);
      if (ownerNameLength > 0 || ownerName.size() > 0) {
        // Usually no owner, maybe this is for quest?
        LOG(INFO) << "Cos picked item with owner \"" << ownerName << "\"(" << ownerNameLength << ")";
      }
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kDropItemCos) {
      LOG(INFO) << "InventoryOperationResponse type kDropItemCos";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kBuyItemCos) {
      LOG(INFO) << "InventoryOperationResponse type kBuyItemCos";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kSellItemCos) {
      LOG(INFO) << "InventoryOperationResponse type kSellItemCos";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kAddCositemByServer) {
      LOG(INFO) << "InventoryOperationResponse type kAddCositemByServer";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kDelCositemByServer) {
      LOG(INFO) << "InventoryOperationResponse type kDelCositemByServer";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kBuyCashItem) {
      LOG(INFO) << "InventoryOperationResponse type kBuyCashItem";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kMoveItemTradeNow) {
      LOG(INFO) << "InventoryOperationResponse type kMoveItemTradeNow";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kPushItemIntoMagicCube) {
      LOG(INFO) << "InventoryOperationResponse type kPushItemIntoMagicCube";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kPopItemFromMagicCube) {
      LOG(INFO) << "InventoryOperationResponse type kPopItemFromMagicCube";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kDelItemInMagicCube) {
      LOG(INFO) << "InventoryOperationResponse type kDelItemInMagicCube";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kActivateMagicCube) {
      LOG(INFO) << "InventoryOperationResponse type kActivateMagicCube";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kBuyItemWithToken) {
      LOG(INFO) << "InventoryOperationResponse type kBuyItemWithToken";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kPickSpecialItem) {
      LOG(INFO) << "InventoryOperationResponse type kPickSpecialItem";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kPickSpecialItemBySilkpet) {
      LOG(INFO) << "InventoryOperationResponse type kPickSpecialItemBySilkpet";
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kPickSpecialItemByOther) {
      LOG(INFO) << "InventoryOperationResponse type kPickSpecialItemByOther";
    } else {
      LOG(INFO) << "Unhandled item movement case! Type: " << static_cast<int>(primaryItemMovement.type);
    }
    if (!itemMovements_.empty()) {
      // There were secondary item movements added, place the primary item movement at the beginning of the list
      itemMovements_.insert(itemMovements_.begin(), primaryItemMovement);
    } else {
      itemMovements_.push_back(primaryItemMovement);
    }
  } else {
    uint16_t errorCode;
    stream.Read(errorCode);
    if (errorCode != 0x0003) { // UIIT_MSG_STRGERR_INVALID_TARGET
      LOG(INFO) << absl::StreamFormat("Item movement failed! Error code %#06X", errorCode);
    }
  }
}
// TODO: Try to inject a buy packet that buys more than 1 stack of a stackable item
//  Invesitage what it does to the kBuyFromNPC data

} // namespace packet::parsing