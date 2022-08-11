#include "serverAgentInventoryOperationResponse.hpp"

#include "logging.hpp"
#include "packet/parsing/commonParsing.hpp"

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
    if (primaryItemMovement.type == packet::enums::ItemMovementType::kWithinInventory ||
        primaryItemMovement.type == packet::enums::ItemMovementType::kAvatarToInventory ||
        primaryItemMovement.type == packet::enums::ItemMovementType::kInventoryToAvatar) {
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
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kInventoryToStorage ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kStorageToInventory ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kInventoryToGuildStorage ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kGuildStorageToInventory) {
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kInventoryToStorage) {
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kGoldDrop ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kGoldStorageWithdraw ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kGoldStorageDeposit ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kGoldGuildStorageDeposit ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kGoldGuildStorageWithdraw) {
      primaryItemMovement.goldAmount = stream.Read<uint64_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kPickItem) {
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      if (primaryItemMovement.destSlot == structures::ItemMovement::kGoldSlot) {
        // Picked gold
        primaryItemMovement.goldPickAmount = stream.Read<uint32_t>();
      } else {
        // Picked an item
        primaryItemMovement.pickedItem = parseGenericItem(stream, itemData);
      }
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kDropItem) {
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kWithinCos) {
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // COS global ID
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kCosToInventory ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kInventoryToCos) {
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // COS global ID
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kWithinStorage ||
               primaryItemMovement.type == packet::enums::ItemMovementType::kWithinGuildStorage) {
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kBuyFromNPC) {
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
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kCosPickGold) {
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // COS global ID
      primaryItemMovement.destSlot = stream.Read<uint8_t>(); // Gold slot, always 0xFE
      primaryItemMovement.goldPickAmount = stream.Read<uint32_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kSellToNPC) {
      primaryItemMovement.srcSlot = stream.Read<uint8_t>();
      primaryItemMovement.quantity = stream.Read<uint16_t>();
      primaryItemMovement.globalId = stream.Read<uint32_t>(); // NPC global ID
      primaryItemMovement.buybackStackSize = stream.Read<uint8_t>();
    } else if (primaryItemMovement.type == packet::enums::ItemMovementType::kBuyback) {
      primaryItemMovement.destSlot = stream.Read<uint8_t>();
      primaryItemMovement.srcSlot = stream.Read<uint8_t>(); // Shop buyback slot, left is max(buybackStackSize-1), right is 0
      primaryItemMovement.quantity = stream.Read<uint16_t>();
    } else {
      LOG() << "Unhandled item movement case! Type: " << static_cast<int>(primaryItemMovement.type) << '\n';
    }
    if (!itemMovements_.empty()) {
      // There were secondary item movements added, place the primary item movement at the beginning of the list
      itemMovements_.insert(itemMovements_.begin(), primaryItemMovement);
    } else {
      itemMovements_.push_back(primaryItemMovement);
    }
  } else {
    LOG() << "Item movement failed! Dumping data\n";
    LOG() << DumpToString(stream) << '\n';
  }
}
// TODO: Try to inject a buy packet that buys more than 1 stack of a stackable item
//  Invesitage what it does to the kBuyFromNPC data

} // namespace packet::parsing