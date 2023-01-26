#include "useItem.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "logging.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"

#include <stdexcept>

namespace state::machine {

UseItem::UseItem(Bot &bot, sro::scalar_types::StorageIndexType inventoryIndex) : StateMachine(bot), inventoryIndex_(inventoryIndex) {
  stateMachineCreated(kName);

  // Do some quick checks to make sure we have this item and can use it
  const auto &inventory = bot_.selfState().inventory;
  if (!inventory.hasItem(inventoryIndex_)) {
    throw std::runtime_error("Trying use nonexistent item in inventory");
  }
  const auto *item = inventory.getItem(inventoryIndex_);
  const auto *itemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(item);
  if (itemAsExpendable == nullptr) {
    throw std::runtime_error("Item is not an expendable");
  }
  itemTypeId_ = itemAsExpendable->typeData();
  lastKnownQuantity_ = itemAsExpendable->quantity;
}

UseItem::~UseItem() {
  stateMachineDestroyed();
}

void UseItem::onUpdate(const event::Event *event) {
  if (event) {
    if (const auto *itemUseFailedEvent = dynamic_cast<const event::ItemUseFailed*>(event)) {
      if (itemUseFailedEvent->reason == packet::enums::InventoryErrorCode::kWaitForReuseDelay) {
        // LOG() << "ItemUseFailed" << std::string(400,'$') << std::endl;
        waitingForItemToBeUsed_ = false;
      } else {
        // TODO: Handle other cases
      }
    } else if (const auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event)) {
      if (inventoryUpdatedEvent->srcSlotNum && *inventoryUpdatedEvent->srcSlotNum == inventoryIndex_) {
        // This is our item
        if (inventoryUpdatedEvent->destSlotNum) {
          // Item was moved to a new slot, track it
          LOG() << "Item was moved to a new slot " << static_cast<int>(inventoryIndex_) << " to " << static_cast<int>(*inventoryUpdatedEvent->destSlotNum) << std::endl;
          inventoryIndex_ = *inventoryUpdatedEvent->destSlotNum;
          // TODO: If we were waiting for this item to be used, it will probably fail now. Will we find out via some other mechanism in the future?
        } else {
          // Check that item at this inventory slot decreased
          bool looksGood{false};
          const auto *item = bot_.selfState().inventory.getItem(*inventoryUpdatedEvent->srcSlotNum);
          if (const auto *itemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(item)) {
            if (itemAsExpendable->quantity == lastKnownQuantity_-1) {
              // Item count decreased by 1, this is the usual expected case
              looksGood = true;
            } else {
              // Item count didnt decrease by 1, not going to consider this a valid item use
              LOG() << "Item changed count from " << lastKnownQuantity_ << " to " << itemAsExpendable->quantity << std::endl;
            }
          } else {
            // The item is now null
            if (lastKnownQuantity_ == 1) {
              // The last item disappeared, that's a successful use
              looksGood = true;
            } else {
              // More than 1 (or 0) items disappeared, that's unusual
              LOG() << "Item is null, and there were " << lastKnownQuantity_ << " remaining" << std::endl;
            }
          }
          if (looksGood) {
            waitingForItemToBeUsed_ = false;
            done_ = true;
            return;
          }
        }
      }
    }
  }
  
  if (waitingForItemToBeUsed_) {
    return;
  }

  // Use item
  const auto itemUsePacket = packet::building::ClientAgentInventoryItemUseRequest::packet(inventoryIndex_, itemTypeId_);
  bot_.packetBroker().injectPacket(itemUsePacket, PacketContainer::Direction::kClientToServer);
  waitingForItemToBeUsed_ = true;
}

bool UseItem::done() const {
  return done_;
}

} // namespace state::machine