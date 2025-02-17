#include "useItem.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"

#include <absl/log/log.h>

#include <stdexcept>

namespace state::machine {

UseItem::UseItem(Bot &bot, sro::scalar_types::StorageIndexType inventoryIndex) : StateMachine(bot), inventoryIndex_(inventoryIndex) {
  stateMachineCreated(kName);

  // Do some quick checks to make sure we have this item and can use it
  const auto &inventory = bot_.selfState()->inventory;
  if (!inventory.hasItem(inventoryIndex_)) {
    throw std::runtime_error("Trying use nonexistent item in inventory");
  }
  const auto *item = inventory.getItem(inventoryIndex_);
  const auto *itemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(item);
  if (itemAsExpendable == nullptr) {
    throw std::runtime_error("Item is not an expendable");
  }
  itemTypeId_ = itemAsExpendable->typeId();
  lastKnownQuantity_ = itemAsExpendable->quantity;
  itemName_ = bot_.gameData().getItemName(item->refItemId);
}

UseItem::~UseItem() {
  if (itemUseTimeoutEventId_) {
    bot_.eventBroker().cancelDelayedEvent(*itemUseTimeoutEventId_);
    itemUseTimeoutEventId_.reset();
  }
  stateMachineDestroyed();
}

Status UseItem::onUpdate(const event::Event *event) {
  if (event) {
    // We don't care about any event if we haven't yet used an item
    if (const auto *itemUseFailedEvent = dynamic_cast<const event::ItemUseFailed*>(event)) {
      if (itemUseTimeoutEventId_) {
          bot_.eventBroker().cancelDelayedEvent(*itemUseTimeoutEventId_);
          itemUseTimeoutEventId_.reset();
        if (itemUseFailedEvent->reason == packet::enums::InventoryErrorCode::kItemDoesNotExist) {
          LOG(INFO) << "Failed to use item because it doesnt exist";
          return Status::kDone;
        } else if (itemUseFailedEvent->reason == packet::enums::InventoryErrorCode::kCharacterDead) {
          LOG(INFO) << "Failed to use item because we're dead";
          return Status::kDone;
        } else if (itemUseFailedEvent->reason == packet::enums::InventoryErrorCode::kWaitForReuseDelay) {
        } else {
          LOG(INFO) << "Failed to use item because of unknown reason";
        }
      } else {
        // TODO: This can happen if an item use failure causes us to exit out of here, construct another UseItem, and enter this function with the same event
      }
    } else if (const auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event)) {
      if (inventoryUpdatedEvent->globalId == bot_.selfState()->globalId) {
        if (inventoryUpdatedEvent->srcSlotNum && *inventoryUpdatedEvent->srcSlotNum == inventoryIndex_) {
          // This is our item
          if (inventoryUpdatedEvent->destSlotNum) {
            // Item was moved to a new slot, track it
            inventoryIndex_ = *inventoryUpdatedEvent->destSlotNum;
            if (itemUseTimeoutEventId_) {
              // We need to cancel the existing timeout event (because it has the wrong inventory slot) and send a new item use timeout event with the updated inventory slot.
              const auto eventEndTime = bot_.eventBroker().delayedEventEndTime(*itemUseTimeoutEventId_);
              if (!eventEndTime) {
                throw std::runtime_error("This item use timeout event does not exist");
              }
              bot_.eventBroker().cancelDelayedEvent(*itemUseTimeoutEventId_);
              itemUseTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent<event::ItemUseTimeout>(*eventEndTime, inventoryIndex_, itemTypeId_);
            }
          } else if (itemUseTimeoutEventId_) {
            // Check that item at this inventory slot decreased
            bool looksGood{false};
            const auto *item = bot_.selfState()->inventory.getItem(*inventoryUpdatedEvent->srcSlotNum);
            if (const auto *itemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(item)) {
              if (itemAsExpendable->quantity == lastKnownQuantity_-1) {
                // Item count decreased by 1, this is the usual expected case
                looksGood = true;
              } else {
                // Item count didnt decrease by 1, not going to consider this a valid item use
                {
                  // TODO: This can happen if we send two requests before we get the first response. If we timeout and the first response was just slow.
                  // TODO:  Solve this issue and put the exception back.
                  looksGood = true;
                  LOG(INFO) << "Item changed count from " << lastKnownQuantity_ << " to " << itemAsExpendable->quantity;
                }
                // throw std::runtime_error("Item changed count from " + std::to_string(lastKnownQuantity_) + " to " + std::to_string(itemAsExpendable->quantity));
              }
            } else {
              // The item is now null
              if (lastKnownQuantity_ == 1) {
                // The last item disappeared, that's a successful use
                looksGood = true;
              } else {
                // More than 1 (or could have been 0) items disappeared, that's unusual
                throw std::runtime_error("Item is null, and there were " + std::to_string(lastKnownQuantity_) + " remaining");
              }
            }
            if (looksGood) {
              // Successfully used the item.
              if (!itemUseTimeoutEventId_) {
                throw std::runtime_error("Didn't have a item use timeout event");
              }
              if (itemUseTimeoutEventId_) {
                bot_.eventBroker().cancelDelayedEvent(*itemUseTimeoutEventId_);
                itemUseTimeoutEventId_.reset();
              }
              return Status::kDone;
            }
          }
        }
      }
    } else if (const auto *itemUseTimeout = dynamic_cast<const event::ItemUseTimeout*>(event)) {
      if (itemUseTimeoutEventId_) {
        if (itemUseTimeout->slotNum == inventoryIndex_ && itemUseTimeout->typeData == itemTypeId_) {
          itemUseTimeoutEventId_.reset();
        } else {
          // Item use timed out, but it's not for the item that we're waiting on.
        }
      }
    }
  }

  if (itemUseTimeoutEventId_) {
    return Status::kNotDone;
  }

  if (!bot_.selfState()->canUseItem(itemTypeId_)) {
    throw std::runtime_error("Cannot use item. How did we get here?");
  }

  // Use item
  const auto itemUsePacket = packet::building::ClientAgentInventoryItemUseRequest::packet(inventoryIndex_, itemTypeId_);
  bot_.packetBroker().injectPacket(itemUsePacket, PacketContainer::Direction::kClientToServer);

  // Create a delayed event that will trigger if our item never gets used.
  itemUseTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent<event::ItemUseTimeout>(std::chrono::milliseconds(kItemUseTimeoutMs), inventoryIndex_, itemTypeId_);
  return Status::kNotDone;
}

} // namespace state::machine