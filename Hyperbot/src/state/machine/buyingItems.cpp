#include "buyingItems.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"
#include "packet/building/serverAgentInventoryOperationResponse.hpp"

namespace state::machine {

BuyingItems::BuyingItems(Bot &bot, const std::map<uint32_t, PurchaseRequest> &itemsToBuy) : StateMachine(bot), itemsToBuy_(itemsToBuy) {
  stateMachineCreated(kName);
  // We must be talking to an NPC at this point
  // Prevent the client from closing the talk dialog
  pushBlockedOpcode(packet::Opcode::kClientAgentActionDeselectRequest);
  // Prevent the client from moving items in inventory
  pushBlockedOpcode(packet::Opcode::kClientAgentInventoryOperationRequest);
}

BuyingItems::~BuyingItems() {
  stateMachineDestroyed();
}

void BuyingItems::onUpdate(const event::Event *event) {
  if (event) {
    if (auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event)) {
      if (inventoryUpdatedEvent->destSlotNum) {
        // TODO: We dont actually know if this was our purchase, for now, we assume it was
        if (inventoryUpdatedEvent->srcSlotNum) {
          // This was a stacking
          waitingOnItemMovementResponse_ = false;
        } else {
          // TODO: Make sure this was our purchase
          //  This could have been a result from another method of aquiring an item.
          //  ex. A pickup by a party member

          // Purchase was successful. Adjust shopping list to reflect the newly desired quantity
          const auto *itemAtInventorySlot = bot_.selfState()->inventory.getItem(*inventoryUpdatedEvent->destSlotNum);
          if (itemAtInventorySlot == nullptr) {
            throw std::runtime_error("Got an item from our inventory, but there's nothing here");
          }
          auto it = itemsToBuy_.find(itemAtInventorySlot->refItemId);
          if (it == itemsToBuy_.end()) {
            throw std::runtime_error("Thought we bought an item, but its not in our to-buy list");
          }
          const auto beforeCount = it->second.quantity;
          uint16_t countBought;
          if (const auto *itemExp = dynamic_cast<const storage::ItemExpendable*>(itemAtInventorySlot)) {
            countBought = itemExp->quantity;
          } else {
            countBought = 1;
          }
          if (countBought > it->second.quantity) {
            throw std::runtime_error("Somehow bought more than we wanted to");
          }
          it->second.quantity -= std::min(countBought, it->second.quantity);

          if (it->second.quantity == 0) {
            // No more of these to buy, delete from shopping list
            itemsToBuy_.erase(it);
          }
          waitingOnBuyResponse_ = false;

          // We successfully blocked the server's purchase response from reaching the client, unblock that packet type
          bot_.proxy().unblockOpcode(packet::Opcode::kServerAgentInventoryOperationResponse);
          // Since we blocked the packet which tells the client about this purchase, we need to spoof an item spawning in the character's inventory
          const auto itemBuySpoofPacket = packet::building::ServerAgentInventoryOperationResponse::addItemByServerPacket(*inventoryUpdatedEvent->destSlotNum, *itemAtInventorySlot);
          bot_.packetBroker().injectPacket(itemBuySpoofPacket, PacketContainer::Direction::kServerToClient);
        }

        if (bot_.selfState()->inventory.hasItem(*inventoryUpdatedEvent->destSlotNum)) {
          // Now, lets see if we want to stack this item. It could have been just bought, or we just stacked some of it into another slot
          const auto *itemAtInventorySlot = bot_.selfState()->inventory.getItem(*inventoryUpdatedEvent->destSlotNum);
          if (itemAtInventorySlot == nullptr) {
            throw std::runtime_error("Got an item from our inventory, but there's nothing here");
          }
          if (const auto *destItemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(itemAtInventorySlot)) {
            const auto refIdToStack = itemAtInventorySlot->refItemId;
            auto inventorySlotsWithThisItem = bot_.selfState()->inventory.findItemsWithRefId(refIdToStack);
            if (inventorySlotsWithThisItem.size() > 1) {
              // Try to stack backwards
              std::reverse(inventorySlotsWithThisItem.begin(), inventorySlotsWithThisItem.end());
              bool stackedAnItem{false};
              for (int i=0; i<inventorySlotsWithThisItem.size(); ++i) {
                const auto laterItemIndex = inventorySlotsWithThisItem[i];
                for (int j=i+1; j<inventorySlotsWithThisItem.size(); ++j) {
                  const auto earlierItemIndex = inventorySlotsWithThisItem[j];
                  if (!bot_.selfState()->inventory.hasItem(earlierItemIndex)) {
                    throw std::runtime_error("We were told there was an item here");
                  }
                  const auto *earlierItem = bot_.selfState()->inventory.getItem(earlierItemIndex);
                  const auto *earlierItemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(earlierItem);
                  if (earlierItemAsExpendable == nullptr) {
                    throw std::runtime_error("This item must be an expendable");
                  }
                  if (earlierItemAsExpendable->quantity == earlierItemAsExpendable->itemInfo->maxStack) {
                    // Stack is already full, cannot stack to here
                    continue;
                  }
                  // We can stack the item in slot laterItemIndex to slot earlierItemIndex
                  const auto spaceLeftInStack = earlierItemAsExpendable->itemInfo->maxStack - earlierItemAsExpendable->quantity;
                  if (!bot_.selfState()->inventory.hasItem(laterItemIndex)) {
                    throw std::runtime_error("We were told there was an item here");
                  }
                  const auto *laterItem = bot_.selfState()->inventory.getItem(laterItemIndex);
                  const auto *laterItemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(laterItem);
                  if (laterItemAsExpendable == nullptr) {
                    throw std::runtime_error("This item must be an expendable");
                  }
                  // Stack item
                  const auto moveItemInInventoryPacket = packet::building::ClientAgentInventoryOperationRequest::withinInventoryPacket(laterItemIndex, earlierItemIndex, std::min(laterItemAsExpendable->quantity, static_cast<uint16_t>(spaceLeftInStack)));
                  bot_.packetBroker().injectPacket(moveItemInInventoryPacket, PacketContainer::Direction::kClientToServer);
                  waitingOnItemMovementResponse_ = true;
                  stackedAnItem = true;
                  break;
                }
                if (stackedAnItem) {
                  // Dont try stacking multiple in one go
                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  if (waitingOnItemMovementResponse_) {
    // Waiting on a stacking, nothing to do
    return;
  }

  if (waitingOnBuyResponse_) {
    // Waiting on an item we bought, nothing to do
    return;
  }

  if (itemsToBuy_.empty()) {
    // Nothing else to buy
    done_ = true;
    return;
  }

  const auto &nextPurchaseRequest = itemsToBuy_.begin()->second;
  const auto countToBuy = std::min(nextPurchaseRequest.quantity, static_cast<uint16_t>(nextPurchaseRequest.maxStackSize));
  // Block the server's response from reaching the client
  bot_.proxy().blockOpcode(packet::Opcode::kServerAgentInventoryOperationResponse);
  const auto buyItemPacket = packet::building::ClientAgentInventoryOperationRequest::buyPacket(nextPurchaseRequest.tabIndex, nextPurchaseRequest.itemIndex, countToBuy, bot_.selfState()->talkingGidAndOption->first);
  bot_.packetBroker().injectPacket(buyItemPacket, PacketContainer::Direction::kClientToServer);
  waitingOnBuyResponse_ = true;
}

bool BuyingItems::done() const {
  return done_;
}

} // namespace state::machine